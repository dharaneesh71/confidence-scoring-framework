import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AreaChart, Area, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import {
  Paper, TextField, Button, Typography, Box,
  CircularProgress, Alert, Chip, Card, CardContent, LinearProgress,
  useTheme, Divider, Rating, Collapse, Skeleton,
  Tooltip, FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import {
  Send, Lightbulb, CheckCircle, Public, Person, Warning
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import '../styles/QAPage.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const isGeneralKnowledge = (res) => res && res.confidence_score <= 0.05;
const isLowConfidence    = (res) => res && res.confidence_score > 0.05 && res.confidence_score < 0.5;

// ── AI MESSAGE BUBBLE ─────────────────────────────────────────────────────────
const AIMessageBubble = ({ msg, user, theme, cardStyle }) => {
  const [rating, setRating]                   = useState(0);
  const [comment, setComment]                 = useState('');
  const [feedbackSent, setFeedbackSent]       = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackError, setFeedbackError]     = useState(null);

  // ── animate confidence bar in on mount ──────────────────────────────────
  const [barValue, setBarValue] = useState(0);
  useEffect(() => {
    // Small delay so the transition is visible after render
    const t = setTimeout(() => {
      if (msg.confidence_score !== undefined && msg.confidence_score !== null) {
        setBarValue(msg.confidence_score * 100);
      }
    }, 150);
    return () => clearTimeout(t);
  }, [msg.confidence_score]);

  const flagged = isLowConfidence(msg);

  const getConfidenceColor = (score) => {
    if (score >= 0.75) return 'success';
    if (score >= 0.5)  return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (score) => {
    if (score >= 0.75) return 'HIGH';
    if (score >= 0.5)  return 'MEDIUM';
    return 'LOW';
  };

  const submitFeedback = async () => {
    if (!msg.history_id) {
      setFeedbackError('Cannot submit feedback: history_id is missing from the API response.');
      return;
    }
    setFeedbackLoading(true);
    setFeedbackError(null);
    try {
      const response = await fetch(`${API_BASE}/api/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}`
        },
        body: JSON.stringify({ history_id: msg.history_id, rating, comment }),
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }
      setFeedbackSent(true);
    } catch (err) {
      setFeedbackError(err.message || 'Failed to submit feedback.');
    } finally {
      setFeedbackLoading(false);
    }
  };

  const references    = msg.citations || msg.sources || msg.source_documents || [];
  const hasReferences = !isGeneralKnowledge(msg) && references.length > 0;

  return (
    <Paper sx={{
      ...cardStyle, mt: 2, mb: 4, textAlign: 'left', width: '100%', maxWidth: '100%',
      border: flagged ? `2px solid ${theme.palette.error.main}` : undefined,
    }}>

      {/* ── 1. Low confidence warning banner ─────────────────────────────── */}
      {flagged && (
        <Box sx={{
          display: 'flex', alignItems: 'center', gap: 1.5, px: 3, py: 1.5,
          bgcolor: theme.palette.mode === 'dark'
            ? 'rgba(211,47,47,0.15)'
            : 'rgba(211,47,47,0.08)',
          borderBottom: `1px solid ${theme.palette.error.main}`,
          borderRadius: '14px 14px 0 0',
        }}>
          <Warning sx={{ color: 'error.main', fontSize: 20 }} />
          <Typography variant="body2" sx={{ color: 'error.main', fontWeight: 'bold' }}>
            LOW CONFIDENCE — This answer may be unreliable. Please verify with official sources.
          </Typography>
        </Box>
      )}

      {/* ── Answer text ──────────────────────────────────────────────────── */}
      <Box sx={{ mb: 3 }}>
        <Typography
          variant="h6"
          color="secondary"
          gutterBottom
          sx={{ display: 'flex', alignItems: 'center', gap: 1 }}
        >
          <Lightbulb fontSize="small" /> Answer:
        </Typography>
        <Box sx={{
          lineHeight: 1.8, fontSize: '1.05rem', pl: 1,
          '& p':       { margin: '0 0 1rem 0' },
          '& ul, & ol': { margin: '0 0 1rem 0', paddingLeft: '2rem' }
        }}>
          <ReactMarkdown>{msg.answer}</ReactMarkdown>
        </Box>
      </Box>

      <Divider sx={{ my: 3 }} />

      {/* ── Confidence block ─────────────────────────────────────────────── */}
      <Box sx={{
        my: 3, p: 3,
        border: `1px solid ${theme.palette.divider}`,
        borderRadius: '16px',
        bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)'
      }}>
        {isGeneralKnowledge(msg) ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Public sx={{ fontSize: 36, color: 'info.main' }} />
            <Box>
              <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'info.main' }}>
                General Knowledge Response
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Answer not found in Ground Truth documents.
              </Typography>
            </Box>
          </Box>
        ) : (
          <>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <CheckCircle fontSize="medium" color="success" />
                <Typography variant="h6" sx={{ fontWeight: 'bold' }}>Verified from Documents</Typography>
              </Box>
              <Chip
                label={`${getConfidenceLabel(msg.confidence_score)} CONFIDENCE`}
                color={getConfidenceColor(msg.confidence_score)}
                sx={{ fontWeight: 'bold' }}
              />
            </Box>

            {/* ── 3. Animated confidence bar ──────────────────────────────── */}
            <LinearProgress
              variant="determinate"
              value={barValue}
              color={getConfidenceColor(msg.confidence_score)}
              sx={{
                height: 10,
                borderRadius: 5,
                mb: 2,
                transition: 'all 1s ease-in-out',
                '& .MuiLinearProgress-bar': {
                  transition: 'transform 1.2s cubic-bezier(0.4, 0, 0.2, 1)',
                },
              }}
            />

            {msg.explanation && (
              <Box sx={{ p: 2, bgcolor: theme.palette.action.hover, borderRadius: '8px' }}>
                <Typography variant="body2" color="text.secondary">
                  <strong>Evaluation Details:</strong> {msg.explanation}
                </Typography>
              </Box>
            )}
          </>
        )}
      </Box>

      {/* ── Source references ─────────────────────────────────────────────── */}
      {hasReferences && (
        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 'bold', color: 'text.secondary' }}>
            SOURCE REFERENCES
          </Typography>
          {references.map((ref, idx) => {
            const sourceName  = ref.source || (ref.metadata && ref.metadata.source) || 'Document';
            const excerptText = ref.excerpt || ref.text || ref.page_content || ref.content || '';
            return (
              <Card key={idx} variant="outlined" sx={{ mb: 1.5, bgcolor: 'transparent' }}>
                <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Typography variant="body2" color="secondary" sx={{ fontWeight: 'bold' }}>
                    {sourceName}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, fontStyle: 'italic' }}>
                    "{excerptText}"
                  </Typography>
                </CardContent>
              </Card>
            );
          })}
        </Box>
      )}

      <Divider sx={{ my: 4 }} />

      {/* ── 5. Inline feedback (per-message) ─────────────────────────────── */}
      <Box sx={{ textAlign: 'center' }}>
        {!feedbackSent ? (
          <>
            <Typography component="legend" sx={{ mb: 1 }}>Rate this answer</Typography>
            <Rating
              value={rating}
              size="large"
              onChange={(e, v) => { setRating(v); setFeedbackError(null); }}
            />
            <Collapse in={rating > 0 && rating < 5}>
              <TextField
                fullWidth multiline rows={2}
                placeholder="What was missing? (Optional)"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                sx={{ mt: 2 }}
              />
            </Collapse>
            {feedbackError && (
              <Alert severity="error" sx={{ mt: 2, borderRadius: '8px', textAlign: 'left' }}>
                {feedbackError}
              </Alert>
            )}
            {rating > 0 && (
              <Button
                onClick={submitFeedback}
                variant="outlined"
                disabled={feedbackLoading}
                sx={{ mt: 2, minWidth: '160px' }}
              >
                {feedbackLoading ? <CircularProgress size={20} /> : 'Submit Feedback'}
              </Button>
            )}
          </>
        ) : (
          <Alert severity="success" sx={{ justifyContent: 'center' }}>
            Thanks for your feedback! ✅
          </Alert>
        )}
      </Box>
    </Paper>
  );
};

// ── MAIN QA PAGE ──────────────────────────────────────────────────────────────
const QAPage = ({ clearSelection }) => {
  const { sessionId } = useParams();
  const navigate      = useNavigate();
  const theme         = useTheme();
  const { user }      = useAuth();

  const [question, setQuestion]                 = useState('');
  const [loading, setLoading]                   = useState(false);
  const [error, setError]                       = useState(null);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages]                 = useState([]);

  // ── Domain state ───────────────────────────────────────────────────────────
  const [domains, setDomains]               = useState([]);
  const [selectedDomain, setSelectedDomain] = useState('');   // '' = All

  const inputBarRef                         = useRef(null);
  const [inputBarHeight, setInputBarHeight] = useState(120);
  const lastQuestionRef                     = useRef(null);
  const chatEndRef                          = useRef(null);

  const cardStyle = {
    p: 4, borderRadius: '16px',
    background: theme.palette.mode === 'dark' ? 'rgba(30, 41, 59, 0.8)' : '#ffffff',
    boxShadow: theme.palette.mode === 'dark'
      ? '0 8px 32px rgba(0,0,0,0.3)'
      : '0 8px 32px rgba(0,0,0,0.05)',
  };

  const chartData = messages
    .filter(m => m.confidence_score !== undefined && m.confidence_score !== null)
    .map((m, index) => ({ turn: index + 1, score: Math.round(m.confidence_score * 100) }));

  // ── Fetch available domains on mount ──────────────────────────────────────
  useEffect(() => {
    if (!user) return;
    fetch(`${API_BASE}/api/domains`, {
      headers: { 'Authorization': `Bearer ${user.token}` }
    })
      .then(res => res.json())
      .then(data => setDomains(data))
      .catch(err => console.error('Error fetching domains', err));
  }, [user]);

  // ── Track input bar height ────────────────────────────────────────────────
  useEffect(() => {
    if (!inputBarRef.current) return;
    const observer = new ResizeObserver(() => {
      if (inputBarRef.current) setInputBarHeight(inputBarRef.current.offsetHeight);
    });
    observer.observe(inputBarRef.current);
    return () => observer.disconnect();
  }, []);

  // ── Scroll behaviour ──────────────────────────────────────────────────────
  useEffect(() => {
    if (loading) {
      setTimeout(() => lastQuestionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50);
    } else {
      chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [loading, messages]);

  // ── Load session or start fresh ───────────────────────────────────────────
  useEffect(() => {
    if (sessionId) { loadSession(sessionId); }
    else           { handleStartNew(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const loadSession = async (id) => {
    setLoading(true); setError(null);
    try {
      const response = await fetch(`${API_BASE}/api/session/${id}`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (!response.ok) throw new Error('Session not found');
      const data = await response.json();
      setCurrentSessionId(data.session_id);
      setMessages(data.messages || []);
    } catch (err) {
      setError('Could not retrieve session details.');
    } finally {
      setLoading(false);
    }
  };

  const handleStartNew = () => {
    setCurrentSessionId(null);
    setMessages([]);
    setQuestion('');
    setError(null);
    navigate('/chat');
    if (clearSelection) clearSelection();
  };

  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!question.trim()) { setError('Please enter a question'); return; }

    const userQuestion = question;
    setQuestion('');
    setError(null);

    // Optimistic update — show user bubble immediately
    setMessages(prev => [...prev, { question: userQuestion, is_optimistic: true }]);
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}`
        },
        body: JSON.stringify({
          question:   userQuestion,
          session_id: currentSessionId,
          domain:     selectedDomain || undefined,   // undefined = all domains
        }),
      });

      if (!response.ok) throw new Error('Failed to get answer');
      const data = await response.json();

      if (!currentSessionId && data.session_id) {
        setCurrentSessionId(data.session_id);
        navigate(`/chat/${data.session_id}`);
      }

      // Replace optimistic message with real response
      setMessages(prev => [...prev.filter(m => !m.is_optimistic), data]);

    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.filter(m => !m.is_optimistic));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ position: 'relative', height: '100%', width: '100%', overflow: 'hidden' }}>

      {/* ── SCROLLABLE CHAT AREA ─────────────────────────────────────────── */}
      <Box sx={{
        position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
        overflowY: 'auto',
        pt: { xs: '20px', sm: '24px' },
        pb: `${inputBarHeight + 16}px`,
        px: { xs: 2, md: 4 },
      }}>
        <Box sx={{ width: '100%', maxWidth: '900px', margin: '0 auto', display: 'flex', flexDirection: 'column' }}>

          {/* Trust score trend chart */}
          {chartData.length > 1 && (
            <Paper sx={{
              p: 2, mb: 4,
              bgcolor: 'rgba(0,209,255,0.05)',
              border: '1px solid rgba(0,209,255,0.2)',
              borderRadius: '12px'
            }}>
              <Typography variant="caption" sx={{ color: '#00d1ff', fontWeight: 'bold' }}>
                SESSION TRUST SCORE TREND (%)
              </Typography>
              <Box sx={{ height: 80, width: '100%', mt: 1 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%"  stopColor="#00d1ff" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#00d1ff" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="score"
                      stroke="#00d1ff"
                      fillOpacity={1}
                      fill="url(#colorScore)"
                      strokeWidth={2}
                    />
                    {/* aliased to RechartsTooltip to avoid conflict with MUI Tooltip */}
                    <RechartsTooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: 'none',
                        borderRadius: '8px',
                        color: '#fff'
                      }}
                    />
                    <YAxis hide domain={[0, 100]} />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </Paper>
          )}

          {/* Empty state */}
          {messages.length === 0 && !loading && (
            <Box sx={{ m: 'auto', textAlign: 'center', color: 'text.secondary', mt: 4 }}>
              <Lightbulb sx={{ fontSize: 60, mb: 2, opacity: 0.5 }} />
              <Typography variant="h6">How can I help you today?</Typography>
              {selectedDomain && (
                <Chip
                  label={`Searching in: ${selectedDomain}`}
                  color="primary" variant="outlined" size="small" sx={{ mt: 1 }}
                />
              )}
            </Box>
          )}

          {/* Messages loop */}
          {messages.map((msg, idx) => {
            const isLastMessage = idx === messages.length - 1;
            return (
              <Box key={idx} sx={{ width: '100%', mb: 2 }}>
                {/* User bubble */}
                {msg.question && (
                  <Box
                    ref={isLastMessage ? lastQuestionRef : null}
                    sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}
                  >
                    <Paper sx={{
                      p: 2,
                      bgcolor: theme.palette.primary.main,
                      color: 'white',
                      borderRadius: '16px 16px 0 16px',
                      maxWidth: '85%'
                    }}>
                      <Typography variant="body1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {msg.question} <Person fontSize="small" />
                      </Typography>
                    </Paper>
                  </Box>
                )}

                {/* AI answer bubble */}
                {msg.answer && !msg.is_optimistic && (
                  <AIMessageBubble msg={msg} user={user} theme={theme} cardStyle={cardStyle} />
                )}
              </Box>
            );
          })}

          {/* ── 1. Loading skeleton (replaces plain spinner) ─────────────── */}
          {loading && (
            <Box sx={{ width: '100%', mb: 4 }}>
              {/* Skeleton mimics the AI bubble layout */}
              <Paper sx={{
                ...cardStyle, mt: 2,
                border: `1px solid ${theme.palette.divider}`,
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                  <Skeleton variant="circular" width={24} height={24} />
                  <Skeleton variant="text" width="20%" height={28} />
                </Box>
                <Skeleton variant="text" width="90%" height={22} />
                <Skeleton variant="text" width="80%" height={22} sx={{ mt: 0.5 }} />
                <Skeleton variant="text" width="85%" height={22} sx={{ mt: 0.5 }} />
                <Skeleton
                  variant="rectangular"
                  height={120}
                  sx={{ mt: 2, borderRadius: 2 }}
                />
                <Skeleton variant="text" width="40%" height={20} sx={{ mt: 1.5 }} />
                <Skeleton variant="text" width="60%" height={20} sx={{ mt: 0.5 }} />
              </Paper>
            </Box>
          )}

          <div ref={chatEndRef} style={{ height: '24px' }} />
        </Box>
      </Box>

      {/* ── FIXED BOTTOM INPUT BAR ───────────────────────────────────────── */}
      <Box
        ref={inputBarRef}
        sx={{
          position: 'fixed', bottom: 0, left: 0, right: 0,
          zIndex: (t) => t.zIndex.appBar - 1,
          bgcolor: theme.palette.mode === 'dark' ? '#0f172a' : '#f5f5f5',
          borderTop: `1px solid ${theme.palette.divider}`,
          pt: 1.5, pb: 3,
          display: 'flex', justifyContent: 'center',
        }}
      >
        <Box sx={{ width: '100%', maxWidth: '900px', px: { xs: 2, md: 4 } }}>

          {/* ── 4. Domain dropdown ──────────────────────────────────────────── */}
          {domains.length > 0 && (
            <Box sx={{ mb: 1.5, display: 'flex', alignItems: 'center', gap: 2 }}>
              <FormControl size="small" sx={{ minWidth: 180 }}>
                <InputLabel id="domain-label">Domain</InputLabel>
                <Select
                  labelId="domain-label"
                  value={selectedDomain}
                  onChange={(e) => setSelectedDomain(e.target.value)}
                  label="Domain"
                  sx={{ borderRadius: '12px' }}
                >
                  <MenuItem value="">
                    <em>All Domains</em>
                  </MenuItem>
                  {domains.map((d) => (
                    <MenuItem key={d} value={d}>{d}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              {selectedDomain && (
                <Chip
                  label={`Filtered: ${selectedDomain}`}
                  size="small"
                  color="primary"
                  variant="outlined"
                  onDelete={() => setSelectedDomain('')}
                />
              )}
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2, borderRadius: '12px' }}>{error}</Alert>
          )}

          {/* Input row */}
          <Paper
            elevation={0}
            sx={{
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: '24px', p: 1,
              display: 'flex', alignItems: 'flex-end',
              bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : '#ffffff',
            }}
          >
            <TextField
              fullWidth multiline maxRows={5} variant="standard"
              placeholder={
                selectedDomain
                  ? `Ask a question in ${selectedDomain}...`
                  : 'Ask a question across all domains...'
              }
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
              InputProps={{ disableUnderline: true, sx: { px: 2, py: 1 } }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
              }}
            />

            {/* ── 2. Submit button with Tooltip ───────────────────────────── */}
            <Tooltip
              title={loading ? 'Thinking...' : !question.trim() ? 'Type a question first' : 'Send question'}
              arrow
              placement="top"
            >
              {/* span wrapper required so Tooltip works on a disabled Button */}
              <span>
                <Button
                  variant="contained"
                  color="secondary"
                  disabled={loading || !question.trim()}
                  onClick={handleSubmit}
                  endIcon={
                    loading
                      ? <CircularProgress size={18} color="inherit" />
                      : <Send fontSize="small" />
                  }
                  sx={{
                    height: '48px',
                    minWidth: '110px',
                    borderRadius: '20px',
                    ml: 1, mb: 0.5,
                    fontWeight: 'bold',
                    textTransform: 'none',
                    fontSize: '0.95rem',
                  }}
                >
                  {loading ? 'Thinking...' : 'Ask'}
                </Button>
              </span>
            </Tooltip>
          </Paper>

          <Typography variant="caption" sx={{ display: 'block', textAlign: 'center', color: 'text.secondary', mt: 1 }}>
            Confid.AI may produce inaccurate information. Please verify critical responses.
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default QAPage;