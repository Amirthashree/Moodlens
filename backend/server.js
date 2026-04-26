// MoodLens Backend — Express + MongoDB
// Run: npm install && node server.js

const express    = require('express');
const mongoose   = require('mongoose');
const bcrypt     = require('bcryptjs');
const jwt        = require('jsonwebtoken');
const cors       = require('cors');
require('dotenv').config();

const app = express();

// ── Middleware ──
app.use(cors({ origin: '*' }));
app.use(express.json());

// ── MongoDB Connection ──
mongoose.connect(process.env.MONGO_URI || 'mongodb://localhost:27017/moodlens')
  .then(() => console.log('✅ MongoDB connected'))
  .catch(err => { console.error('❌ MongoDB error:', err.message); process.exit(1); });

// ══════════════════════════════════════
//  MODELS
// ══════════════════════════════════════

const userSchema = new mongoose.Schema({
  name:      { type: String, required: true, trim: true },
  email:     { type: String, required: true, unique: true, lowercase: true, trim: true },
  password:  { type: String, required: true, minlength: 8 },
  createdAt: { type: Date, default: Date.now },
});
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});
userSchema.methods.comparePassword = function(candidate) {
  return bcrypt.compare(candidate, this.password);
};
const User = mongoose.model('User', userSchema);

const analysisSchema = new mongoose.Schema({
  userId:     { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true, index: true },
  text:       { type: String, required: true },
  emotion:    { type: String, enum: ['Happy','Angry','Frustrated','Sad','Excited','Neutral'], required: true },
  confidence: { type: Number, min: 0, max: 1 },
  keywords:   [String],
  timestamp:  { type: Date, default: Date.now },
});
const Analysis = mongoose.model('Analysis', analysisSchema);

// ══════════════════════════════════════
//  HELPERS
// ══════════════════════════════════════

const JWT_SECRET = process.env.JWT_SECRET || 'moodlens-dev-secret-change-in-prod';

function signToken(userId) {
  return jwt.sign({ id: userId }, JWT_SECRET, { expiresIn: '7d' });
}

function authMiddleware(req, res, next) {
  const header = req.headers.authorization;
  if (!header || !header.startsWith('Bearer '))
    return res.status(401).json({ message: 'No token provided.' });
  try {
    const payload = jwt.verify(header.split(' ')[1], JWT_SECRET);
    req.userId = payload.id;
    next();
  } catch {
    return res.status(401).json({ message: 'Invalid or expired token.' });
  }
}

function userResponse(user) {
  return { id: user._id, name: user.name, email: user.email };
}

// ══════════════════════════════════════
//  FLASK HELPER — with retry + timeout
// ══════════════════════════════════════

const FLASK_BASE = (process.env.FLASK_API_URL || 'http://localhost:5001').replace(/\/$/, '');
console.log(`🔗 Flask API base: ${FLASK_BASE}`);

// Calls Flask /predict with a timeout. Retries once after 7s if it fails
// (handles Render free-tier cold start ~30s spin-up)
async function callFlask(text, retries = 1) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000); // 20s timeout

    const response = await fetch(`${FLASK_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
      signal:  controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Flask responded ${response.status}: ${errText}`);
    }

    return await response.json();

  } catch (err) {
    if (err.name === 'AbortError') err = new Error('Flask timed out after 20s');

    if (retries > 0) {
      console.log(`⚠️  Flask call failed (${err.message}), retrying in 7s...`);
      await new Promise(r => setTimeout(r, 7000)); // wait 7s for cold start
      return callFlask(text, retries - 1);
    }

    throw err;
  }
}

// ══════════════════════════════════════
//  KEEP FLASK WARM (free-tier fix)
//  Pings Flask /health every 10 minutes
//  so it never goes to sleep
// ══════════════════════════════════════
function startFlaskKeepAlive() {
  const INTERVAL = 10 * 60 * 1000; // 10 minutes

  const ping = async () => {
    try {
      const r = await fetch(`${FLASK_BASE}/health`, {
        signal: AbortSignal.timeout(10000),
      });
      console.log(`💓 Flask keep-alive ping: ${r.status}`);
    } catch (e) {
      console.log(`⚠️  Flask keep-alive failed: ${e.message}`);
    }
  };

  // Initial ping on startup
  ping();
  setInterval(ping, INTERVAL);
  console.log('⏰ Flask keep-alive started (every 10 min)');
}

// ══════════════════════════════════════
//  AUTH ROUTES
// ══════════════════════════════════════

app.post('/api/auth/register', async (req, res) => {
  try {
    const { name, email, password } = req.body;
    if (!name || !email || !password)
      return res.status(400).json({ message: 'All fields are required.' });
    if (password.length < 8)
      return res.status(400).json({ message: 'Password must be at least 8 characters.' });

    const existing = await User.findOne({ email });
    if (existing)
      return res.status(409).json({ message: 'An account with this email already exists.' });

    const user = await User.create({ name, email, password });
    const token = signToken(user._id);
    res.status(201).json({ token, user: userResponse(user) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error during registration.' });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    if (!email || !password)
      return res.status(400).json({ message: 'Email and password are required.' });

    const user = await User.findOne({ email });
    if (!user || !(await user.comparePassword(password)))
      return res.status(401).json({ message: 'Invalid email or password.' });

    const token = signToken(user._id);
    res.json({ token, user: userResponse(user) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Server error during login.' });
  }
});

app.get('/api/me', authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.userId).select('-password');
    if (!user) return res.status(404).json({ message: 'User not found.' });
    res.json({ user: userResponse(user) });
  } catch (err) {
    res.status(500).json({ message: 'Server error.' });
  }
});

// ══════════════════════════════════════
//  ANALYZE ROUTE
// ══════════════════════════════════════

app.post('/api/analyze', authMiddleware, async (req, res) => {
  try {
    const { text } = req.body;
    if (!text || !text.trim())
      return res.status(400).json({ message: 'Text is required.' });

    let result;
    try {
      result = await callFlask(text.trim());
    } catch (flaskErr) {
      console.error('❌ Flask call failed:', flaskErr.message);
      return res.status(502).json({
        message: `Python model API error: ${flaskErr.message}`,
        flask_url: FLASK_BASE,
      });
    }

    return res.json({
      emotion:    result.emotion,
      confidence: result.confidence,
      keywords:   result.keywords || [],
      model_used: result.model_used || 'unknown',
      text,
      timestamp:  new Date().toISOString(),
    });

  } catch (err) {
    console.error('Analyze error:', err.message);
    return res.status(500).json({ message: 'Internal server error.' });
  }
});

// ══════════════════════════════════════
//  ANALYSIS ROUTES
// ══════════════════════════════════════

app.post('/api/analyses', authMiddleware, async (req, res) => {
  try {
    const { text, emotion, confidence, keywords, timestamp } = req.body;
    const analysis = await Analysis.create({
      userId: req.userId,
      text, emotion, confidence,
      keywords: keywords || [],
      timestamp: timestamp ? new Date(timestamp) : new Date(),
    });
    res.status(201).json({ analysis });
  } catch (err) {
    console.error(err);
    res.status(500).json({ message: 'Failed to save analysis.' });
  }
});

app.get('/api/analyses', authMiddleware, async (req, res) => {
  try {
    const analyses = await Analysis.find({ userId: req.userId })
      .sort({ timestamp: -1 })
      .limit(100)
      .select('-userId -__v');
    res.json({ analyses });
  } catch (err) {
    res.status(500).json({ message: 'Failed to fetch analyses.' });
  }
});

app.delete('/api/analyses', authMiddleware, async (req, res) => {
  try {
    await Analysis.deleteMany({ userId: req.userId });
    res.json({ message: 'History cleared.' });
  } catch (err) {
    res.status(500).json({ message: 'Failed to clear history.' });
  }
});

app.get('/api/analyses/stats', authMiddleware, async (req, res) => {
  try {
    const stats = await Analysis.aggregate([
      { $match: { userId: new mongoose.Types.ObjectId(req.userId) } },
      { $group: { _id: '$emotion', count: { $sum: 1 }, avgConf: { $avg: '$confidence' } } },
    ]);
    const total = await Analysis.countDocuments({ userId: req.userId });
    res.json({ stats, total });
  } catch (err) {
    res.status(500).json({ message: 'Failed to load stats.' });
  }
});

// ══════════════════════════════════════
//  HEALTH CHECK
// ══════════════════════════════════════
app.get('/api/health', async (req, res) => {
  let flaskStatus = 'unreachable';
  try {
    const r = await fetch(`${FLASK_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    if (r.ok) {
      const data = await r.json();
      flaskStatus = data.status === 'ok' ? `ok (model: ${data.model_type})` : 'error';
    }
  } catch {
    flaskStatus = 'unreachable';
  }
  res.json({ status: 'ok', service: 'MoodLens API', flask_base: FLASK_BASE, flask: flaskStatus });
});

// ── Serve frontend ──
const path = require('path');
app.use(express.static(path.join(__dirname, '../frontend')));

// ── Start ──
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🔮 MoodLens API running on http://localhost:${PORT}`);
  startFlaskKeepAlive(); // ← start pinging Flask after server is up
});