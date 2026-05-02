import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AreaChart, Area, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import {
  TextField, Button, Typography, Box,
  CircularProgress, Alert, Chip, Card, CardContent, LinearProgress,
  useTheme, Divider, Rating, Collapse, Skeleton,
  Tooltip, FormControl, InputLabel, Select, MenuItem
} from '@mui/material';
import { Send, Lightbulb, CheckCircle, Public, Warning } from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import '../styles/QAPage.css';
import StarBackground from '../components/Starbackground';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const isGeneralKnowledge = (res) => res && res.confidence_score <= 0.05;
const isLowConfidence = (res) => res && res.confidence_score > 0.05 && res.confidence_score < 0.5;

// ── AI MESSAGE BUBBLE ─────────────────────────────────────────────────────────
const AIMessageBubble = ({ msg, user, theme, cardStyle }) => {
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackError, setFeedbackError] = useState(null);
  const [barValue, setBarValue] = useState(0);

  useEffect(() => {
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
    if (score >= 0.5) return 'warning';
    return 'error';
  };

  const getConfidenceLabel = (score) => {
    if (score >= 0.75) return 'HIGH';
    if (score >= 0.5) return 'MEDIUM';
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

  const references = msg.citations || msg.sources || msg.source_documents || [];
  const hasReferences = !isGeneralKnowledge(msg) && references.length > 0;

  return (
    <Card sx={{ ...cardStyle, mb: 2 }}>
      <CardContent>
        {flagged && (
          <Alert severity="warning" icon={<Warning />} sx={{ mb: 2 }}>
            LOW CONFIDENCE — This answer may be unreliable. Please verify with official sources.
          </Alert>
        )}

<Typography
  variant="overline"
  display="flex"
  alignItems="center"
  gap={0.5}
  sx={{ color: '#38bdf8', fontWeight: 800, letterSpacing: 1.5, fontSize: '0.72rem' }}
>
  <Lightbulb fontSize="small" sx={{ color: '#fbbf24' }} /> Answer:
</Typography>

<Box
  sx={{
    mt: 1,
    '& p': {
      fontSize: '1.05rem',        // ✅ bumped from 0.97rem
      lineHeight: 1.9,            // ✅ slightly more breathing room
      color: theme.palette.mode === 'dark' ? 'rgba(226,232,240,0.92)' : '#1e293b',
      mb: 1.5,
      fontWeight: 400,
      letterSpacing: '0.01em',
    },
    '& strong': { color: '#e2e8f0', fontWeight: 700 },
    '& code': {
      fontFamily: '"Fira Code", "JetBrains Mono", monospace',
      fontSize: '0.93rem',        // ✅ bumped from 0.88rem
      bgcolor: 'rgba(255,255,255,0.07)',
      px: 0.7, py: 0.2,
      borderRadius: '4px',
    },
    '& ul, & ol': { pl: 2.5, lineHeight: 1.9 },
    '& li': { mb: 0.5, fontSize: '1.05rem', color: 'rgba(226,232,240,0.88)' },  // ✅ bumped
  }}
>
  <ReactMarkdown>{msg.answer}</ReactMarkdown>
</Box>


        <Divider sx={{ my: 2 }} />

        {isGeneralKnowledge(msg) ? (
          <Box display="flex" alignItems="center" gap={1}>
            <Public fontSize="small" color="action" />
            <Typography variant="caption" color="text.secondary">General Knowledge Response</Typography>
            <Typography variant="caption" color="text.secondary">— Answer not found in Ground Truth documents.</Typography>
          </Box>
        ) : (
          <>
            <Box display="flex" alignItems="center" gap={1} mb={1}>
              <CheckCircle fontSize="small" color="success" />
              <Typography variant="caption" color="success.main" fontWeight="bold">Verified from Documents</Typography>
              <Chip
                label={`${getConfidenceLabel(msg.confidence_score)} ${Math.round(msg.confidence_score * 100)}%`}
                color={getConfidenceColor(msg.confidence_score)}
                size="small"
                sx={{ ml: 'auto' }}
              />
            </Box>
            <LinearProgress
              variant="determinate"
              value={barValue}
              color={getConfidenceColor(msg.confidence_score)}
              sx={{ height: 6, borderRadius: 3, transition: 'value 0.8s ease' }}
            />
            {msg.explanation && (
              <Typography variant="caption" color="text.secondary" display="block" mt={1}>
                Evaluation Details: {msg.explanation}
              </Typography>
            )}
          </>
        )}

        {hasReferences && (
          <Box mt={2}>
            <Typography variant="overline" color="text.secondary">SOURCE REFERENCES</Typography>
            {references.map((ref, idx) => {
              const sourceName = ref.source || (ref.metadata && ref.metadata.source) || 'Document';
              const excerptText = ref.excerpt || ref.text || ref.page_content || ref.content || '';
              return (
                <Box key={idx} sx={{ mt: 1, p: 1.5, borderRadius: 1, bgcolor: 'action.hover' }}>
                  <Typography variant="caption" fontWeight="bold">{sourceName}</Typography>
                  {excerptText && (
                    <Typography variant="caption" color="text.secondary" display="block">
                      "{excerptText}"
                    </Typography>
                  )}
                </Box>
              );
            })}
          </Box>
        )}

        <Divider sx={{ my: 2 }} />

        {!feedbackSent ? (
          <>
            <Typography variant="caption" color="text.secondary">Rate this answer</Typography>
            <Rating
              value={rating}
              onChange={(_, v) => { setRating(v); setFeedbackError(null); }}
              sx={{ display: 'flex', flexDirection: 'row', mt: 0.5 }}
            />
            <Collapse in={rating > 0}>
              <TextField
                label="Comments (optional)"
                multiline
                rows={2}
                fullWidth
                value={comment}
                onChange={e => setComment(e.target.value)}
                sx={{ mt: 2 }}
                size="small"
              />
            </Collapse>
            {feedbackError && <Alert severity="error" sx={{ mt: 1 }}>{feedbackError}</Alert>}
            {rating > 0 && (
              <Button
                onClick={submitFeedback}
                disabled={feedbackLoading}
                size="small"
                variant="outlined"
                sx={{ mt: 1 }}
              >
                {feedbackLoading ? <CircularProgress size={16} /> : 'Submit Feedback'}
              </Button>
            )}
          </>
        ) : (
          <Typography variant="body2" color="success.main">Thanks for your feedback! ✅</Typography>
        )}
      </CardContent>
    </Card>
  );
};

// ── MAIN QA PAGE ──────────────────────────────────────────────────────────────
const QAPage = ({ clearSelection }) => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const theme = useTheme();
  const { user } = useAuth();

  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [domains, setDomains] = useState([]);
  const [selectedDomain, setSelectedDomain] = useState('');

  const chatEndRef = useRef(null);
  const scrollContainerRef = useRef(null);

  const cardStyle = {
    borderRadius: '16px',
    background: theme.palette.mode === 'dark' ? '#161b27' : '#ffffff',  // ✅ MUST be solid
    boxShadow: theme.palette.mode === 'dark'
      ? '0 2px 16px rgba(0,0,0,0.6), 0 1px 0 rgba(255,255,255,0.05)'
      : '0 8px 32px rgba(0,0,0,0.08)',
    border: theme.palette.mode === 'dark'
      ? '1px solid rgba(255,255,255,0.07)'
      : '1px solid rgba(0,0,0,0.06)',
  };
  
  

  const chartData = messages
    .filter(m => m.confidence_score !== undefined && m.confidence_score !== null)
    .map((m, index) => ({ turn: index + 1, score: Math.round(m.confidence_score * 100) }));

  useEffect(() => {
    if (!user) return;
    fetch(`${API_BASE}/api/domains`, { headers: { 'Authorization': `Bearer ${user.token}` } })
      .then(res => res.json())
      .then(data => setDomains(data))
      .catch(err => console.error('Error fetching domains', err));
  }, [user]);

  // ✅ FIX: reliable scroll — always scroll the container to bottom
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    // Use requestAnimationFrame to ensure DOM has painted before scrolling
    requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
  }, [messages, loading]);

  useEffect(() => {
    if (sessionId) { loadSession(sessionId); }
    else { handleStartNew(); }
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
          question: userQuestion,
          session_id: currentSessionId,
          domain: selectedDomain || undefined,
        }),
      });

      if (!response.ok) throw new Error('Failed to get answer');
      const data = await response.json();

      if (!currentSessionId && data.session_id) {
        setCurrentSessionId(data.session_id);
        navigate(`/chat/${data.session_id}`);
      }
      setMessages(prev => [...prev.filter(m => !m.is_optimistic), data]);
    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.filter(m => !m.is_optimistic));
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Box
  sx={{
    position: 'fixed',
    inset: 0,
    top: '64px',
    display: 'flex',
    flexDirection: 'column',
    bgcolor: theme.palette.mode === 'dark' ? '#000000' : '#f0f4ff',
    overflow: 'hidden',
  }}
>
<StarBackground mode={theme.palette.mode} />
<Box
  ref={scrollContainerRef}
  sx={{
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    position: 'relative',   
    zIndex: 1,         
    bgcolor: 'transparent',
    px: { xs: 1, sm: 2, md: 4 },
    py: 3,
    display: 'flex',
    flexDirection: 'column',
  }}
>
        <Box sx={{ maxWidth: 1050, width: '100%', mx: 'auto', flex: 1, display: 'flex', flexDirection: 'column' }}>
        
          {/* Trust score trend */}
          {chartData.length > 1 && (
            <Card sx={{ ...cardStyle, mb: 3, p: 2 }}>
              <Typography variant="overline" sx={{ color: '#38bdf8', letterSpacing: 1.5, fontSize: '0.7rem', fontWeight: 700 }}>SESSION TRUST SCORE TREND (%)</Typography>
              <ResponsiveContainer width="100%" height={80}>
                <AreaChart data={chartData}>
                  <YAxis domain={[0, 100]} hide />
                  <RechartsTooltip formatter={(v) => [`${v}%`, 'Confidence']} />
                  <Area type="monotone" dataKey="score" stroke="#38bdf8" fill="rgba(56,189,248,0.12)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          )}
          
          {/* ✅ FIX: Empty state — centered, NOT covered by loading skeleton */}
          {messages.length === 0 && !loading && (
            <Box
              sx={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: 300,
                gap: 2,
                opacity: 0.7,
              }}
            >
              <Lightbulb sx={{ fontSize: 48, color: 'primary.main' }} />
              <Typography variant="h6" color="text.secondary">
                How can I help you today?
              </Typography>
              {selectedDomain && (
                <Chip label={`Filtered: ${selectedDomain}`} onDelete={() => setSelectedDomain('')} />
              )}
            </Box>
          )}

          {/* Messages */}
          {messages.map((msg, idx) => (
            <Box key={idx} sx={{ mb: 2 }}>
              {msg.question && (
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
                  <Box
                    sx={{
                      bgcolor: 'primary.main',
                      color: 'primary.contrastText',
                      px: 2.5, py: 1.5,
                      borderRadius: '18px 18px 4px 18px',
                      maxWidth: '75%',
                      wordBreak: 'break-word',
                    }}
                  >
                    <Typography variant="body1">{msg.question}</Typography>
                  </Box>
                </Box>
              )}
              {msg.answer && !msg.is_optimistic && (
                <AIMessageBubble msg={msg} user={user} theme={theme} cardStyle={cardStyle} />
              )}
            </Box>
          ))}

          {/* ✅ FIX: Loading skeleton appears AFTER messages, not over them */}
          {loading && (
            <Card sx={{ ...cardStyle, mb: 2 }}>
              <CardContent>
                <Skeleton variant="text" width="40%" height={24} />
                <Skeleton variant="rectangular" height={80} sx={{ mt: 1, borderRadius: 1 }} />
                <Skeleton variant="text" width="60%" sx={{ mt: 1 }} />
                <Skeleton variant="text" width="80%" />
              </CardContent>
            </Card>
          )}

          {/* Scroll anchor */}
          <div ref={chatEndRef} />
        </Box>
      </Box>

      <Box
  component="form"
  onSubmit={handleSubmit}
  sx={{
    flexShrink: 0,
    position: 'relative',   // ✅ ADD THIS
    zIndex: 1,              // ✅ ADD THIS
    bgcolor: theme.palette.mode === 'dark' ? 'transparent' : 'rgba(255,255,255,0.88)',
    backdropFilter: theme.palette.mode === 'dark' ? 'none' : 'blur(12px)',
    borderTop: `1px solid ${theme.palette.divider}`,
    px: { xs: 1, sm: 2, md: 4 },
    pt: 1,
    pb: 1.5,
  }}
>

        <Box sx={{ maxWidth: 800, mx: 'auto', display: 'flex', flexDirection: 'column', gap: 1 }}>

          {/* Domain row */}
          {domains.length > 0 && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              <FormControl size="small" sx={{ minWidth: 130 }}>
                <InputLabel>Domain</InputLabel>
                <Select
                  value={selectedDomain}
                  onChange={e => setSelectedDomain(e.target.value)}
                  label="Domain"
                  sx={{ borderRadius: '12px' }}
                >
                  <MenuItem value="">All Domains</MenuItem>
                  {domains.map(d => <MenuItem key={d} value={d}>{d}</MenuItem>)}
                </Select>
              </FormControl>
              {selectedDomain && (
                <Chip
                  label={`Filtered: ${selectedDomain}`}
                  onDelete={() => setSelectedDomain('')}
                  size="small"
                />
              )}
            </Box>
          )}

          {error && <Alert severity="error" sx={{ py: 0 }}>{error}</Alert>}

          {/* Input + button row */}
          <Box sx={{ display: 'flex', alignItems: 'flex-end', gap: 1 }}>
            <TextField
              fullWidth
              multiline
              maxRows={4}      // ✅ FIX: limits growth, no more giant textarea
              minRows={1}
              placeholder={selectedDomain ? `Ask a question in ${selectedDomain}...` : 'Ask a question across all domains...'}
              value={question}
              onChange={e => setQuestion(e.target.value)}
              disabled={loading}
              variant="outlined"
              size="small"
              InputProps={{ sx: { borderRadius: '20px', px: 2, py: 0.75, fontSize: '0.95rem' } }}
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
              }}
            />
            <Tooltip title={!question.trim() ? 'Enter a question first' : ''}>
              <span>
                <Button
                  type="submit"
                  variant="contained"
                  disabled={loading || !question.trim()}
                  endIcon={loading ? null : <Send />}
                  sx={{
                    height: '40px',
                    minWidth: '100px',
                    borderRadius: '20px',
                    fontWeight: 'bold',
                    textTransform: 'none',
                    fontSize: '0.9rem',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {loading ? <CircularProgress size={18} color="inherit" /> : 'Ask'}
                </Button>
              </span>
            </Tooltip>
          </Box>

          <Typography variant="caption" color="text.secondary" textAlign="center">
            Confid.AI may produce inaccurate information. Please verify critical responses.
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default QAPage;
