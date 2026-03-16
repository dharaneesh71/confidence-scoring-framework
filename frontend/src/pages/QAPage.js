import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import {
  Container, Paper, TextField, Button, Typography, Box,
  CircularProgress, Alert, Chip, Card, CardContent, LinearProgress,
  useTheme, Divider, Rating, Collapse
} from '@mui/material';
import { 
  Send, Lightbulb, CheckCircle, Public, Person
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import '../styles/QAPage.css';

const isGeneralKnowledge = (res) => res && res.confidence_score <= 0.05;

// --- 1. SUB-COMPONENT: AI MESSAGE BUBBLE ---
const AIMessageBubble = ({ msg, user, theme, cardStyle }) => {
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);

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
    if (!msg.history_id) return;
    try {
      await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}`
        },
        body: JSON.stringify({
          history_id: msg.history_id,
          rating: rating,
          comment: comment
        }),
      });
      setFeedbackSent(true);
    } catch (err) {
      console.error("Feedback failed", err);
    }
  };

  return (
    <Paper className="result-paper-fade" sx={{ ...cardStyle, mt: 2, mb: 4, textAlign: 'left', width: '100%', maxWidth: '100%' }}>
      
      {/* ANSWER SECTION */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" color="secondary" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Lightbulb fontSize="small" /> Answer:
        </Typography>
        <Typography paragraph sx={{ lineHeight: 1.8, fontSize: '1.05rem', pl: 1, whiteSpace: 'pre-wrap' }}>
          {msg.answer}
        </Typography>
      </Box>

      <Divider sx={{ my: 3 }} />
      
      {/* VERIFICATION & CONFIDENCE SECTION */}
      <Box sx={{ my: 3, p: 3, border: `1px solid ${theme.palette.divider}`, borderRadius: '16px', bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)' }}>
        {isGeneralKnowledge(msg) ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Public sx={{ fontSize: 36, color: 'info.main' }} />
                <Box>
                    <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'info.main' }}>General Knowledge Response</Typography>
                    <Typography variant="body2" color="text.secondary">Answer not found in Ground Truth documents.</Typography>
                </Box>
            </Box>
        ) : (
            <>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CheckCircle fontSize="medium" color="success" />
                      <Typography variant="h6" sx={{ fontWeight: 'bold' }}>Verified from Documents</Typography>
                  </Box>
                  <Chip label={`${getConfidenceLabel(msg.confidence_score)} CONFIDENCE`} color={getConfidenceColor(msg.confidence_score)} sx={{ fontWeight: 'bold' }} />
              </Box>
              <LinearProgress variant="determinate" value={msg.confidence_score * 100} color={getConfidenceColor(msg.confidence_score)} sx={{ height: 10, borderRadius: 5, mb: 2 }} />
              {msg.explanation && (
                <Box sx={{ p: 2, bgcolor: theme.palette.action.hover, borderRadius: '8px' }}>
                  <Typography variant="body2" color="text.secondary"><strong>Evaluation Details:</strong> {msg.explanation}</Typography>
                </Box>
              )}
            </>
        )}
      </Box>
      
      {/* CITATIONS SECTION */}
      {!isGeneralKnowledge(msg) && msg.citations && msg.citations.length > 0 && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 'bold', color: 'text.secondary' }}>SOURCE REFERENCES</Typography>
            {msg.citations.map((cit, idx) => (
                <Card key={idx} variant="outlined" sx={{ mb: 1.5, bgcolor: 'transparent' }}>
                    <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                        <Typography variant="body2" color="secondary" sx={{ fontWeight: 'bold' }}>{cit.source}</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, fontStyle: 'italic' }}>"{cit.excerpt}"</Typography>
                    </CardContent>
                </Card>
            ))}
          </Box>
      )}

      <Divider sx={{ my: 4 }} />

      {/* FEEDBACK SECTION */}
      <Box sx={{ textAlign: 'center' }}>
        {!feedbackSent ? (
          <>
            <Typography component="legend" sx={{ mb: 1 }}>Rate this answer</Typography>
            <Rating
              value={rating}
              size="large"
              onChange={(event, newValue) => setRating(newValue)}
            />
            
            <Collapse in={rating > 0 && rating < 5}>
              <TextField
                fullWidth
                multiline
                rows={2}
                placeholder="What was missing? (Optional)"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                sx={{ mt: 2 }}
              />
            </Collapse>

            {rating > 0 && (
              <Button onClick={submitFeedback} variant="outlined" sx={{ mt: 2 }}>
                Submit Feedback
              </Button>
            )}
          </>
        ) : (
          <Alert severity="success" sx={{ justifyContent: 'center' }}>Thanks for your feedback!</Alert>
        )}
      </Box>

    </Paper>
  );
};

// --- 2. MAIN QA PAGE COMPONENT ---
const QAPage = ({ clearSelection }) => {
  const { sessionId } = useParams(); // Task #7: Routing URL params
  const navigate = useNavigate();

  const theme = useTheme();
  const { user } = useAuth();
  
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  
  const chatEndRef = useRef(null);

  const cardStyle = {
    p: 4, 
    borderRadius: '16px',
    background: theme.palette.mode === 'dark' ? 'rgba(30, 41, 59, 0.8)' : '#ffffff',
    boxShadow: theme.palette.mode === 'dark' ? '0 8px 32px rgba(0,0,0,0.3)' : '0 8px 32px rgba(0,0,0,0.05)',
  };

  // Task #5: Format data for Recharts Trend Line
  const chartData = messages
    .filter(m => m.confidence_score !== undefined && m.confidence_score !== null)
    .map((m, index) => ({
      turn: index + 1,
      score: Math.round(m.confidence_score * 100)
    }));

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (sessionId) {
      loadSession(sessionId);
    } else {
      handleStartNew();
    }
  }, [sessionId]);

  const loadSession = async (id) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8000/api/session/${id}`, {
        headers: { 'Authorization': `Bearer ${user.token}` }
      });
      if (!response.ok) throw new Error("Session not found");
      const data = await response.json();
      
      setCurrentSessionId(data.session_id);
      setMessages(data.messages || []);
    } catch (err) {
      setError("Could not retrieve session details.");
    } finally {
      setLoading(false);
    }
  };

  // Task #2: New Chat Logic
  const handleStartNew = () => {
    setCurrentSessionId(null);
    setMessages([]);
    setQuestion('');
    setError(null);
    navigate('/chat');
    if(clearSelection) clearSelection();
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
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}`
        },
        body: JSON.stringify({ 
          question: userQuestion,
          session_id: currentSessionId 
        }),
      });

      if (!response.ok) throw new Error("Failed to get answer");
      const data = await response.json();
      
      if (!currentSessionId && data.session_id) {
        setCurrentSessionId(data.session_id);
        navigate(`/chat/${data.session_id}`); // Route to new ID
      }

      setMessages(prev => {
        const filtered = prev.filter(m => !m.is_optimistic);
        return [...filtered, data];
      });

    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.filter(m => !m.is_optimistic));
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Container 
      maxWidth={false} 
      className="qa-container" 
      sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100vh', 
        maxWidth: '1400px', 
        width: '95%', 
        pt: 18,
        pb: 2 
      }}
    >
    
      {/* HEADER SECTION */}
      <Box sx={{ textAlign: 'center', mb: 2, flexShrink: 0 }}>
        <Typography variant="h1" className="glassy-text" sx={{ mb: 1, fontSize: '3.5rem' }}>
          CONFID.AI
        </Typography>
        <Typography variant="subtitle1" sx={{ color: 'text.secondary', letterSpacing: '1px' }}>
          Continuous Evidence-Based Chat
        </Typography>
      </Box>

      {/* Task #5: TRUST SCORE MINI-GRAPH (RECHARTS) */}
      {chartData.length > 1 && (
        <Paper sx={{ p: 2, mb: 3, mx: { xs: 1, md: 3 }, bgcolor: 'rgba(0, 209, 255, 0.05)', border: '1px solid rgba(0, 209, 255, 0.2)' }}>
          <Typography variant="caption" sx={{ color: '#00d1ff', fontWeight: 'bold' }}>SESSION TRUST SCORE TREND (%)</Typography>
          <Box sx={{ height: 80, width: '100%', mt: 1 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00d1ff" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#00d1ff" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="score" stroke="#00d1ff" fillOpacity={1} fill="url(#colorScore)" strokeWidth={2} />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
                <YAxis hide domain={[0, 100]} />
              </AreaChart>
            </ResponsiveContainer>
          </Box>
        </Paper>
      )}

      {/* CHAT HISTORY AREA (Scrollable) */}
      <Box sx={{ flexGrow: 1, overflowY: 'auto', px: { xs: 1, md: 3 }, mb: 3, display: 'flex', flexDirection: 'column' }}>
        
        {messages.length === 0 && !loading && (
             <Box sx={{ m: 'auto', textAlign: 'center', color: 'text.secondary' }}>
                 <Typography variant="h5">Start a new conversation</Typography>
                 <Typography>Type a question below to begin.</Typography>
             </Box>
        )}

        {messages.map((msg, idx) => (
          <Box key={idx} sx={{ width: '100%', mb: 1 }}>
            
            {msg.question && (
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
                <Paper sx={{ p: 2, bgcolor: theme.palette.primary.main, color: 'white', borderRadius: '16px 16px 0 16px', maxWidth: '85%' }}>
                  <Typography variant="body1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {msg.question} <Person fontSize="small" />
                  </Typography>
                </Paper>
              </Box>
            )}

            {/* AI BUBBLE */}
            {msg.answer && !msg.is_optimistic && (
               <AIMessageBubble msg={msg} user={user} theme={theme} cardStyle={cardStyle} />
            )}
            
          </Box>
        ))}

        {loading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', my: 4 }}>
            <CircularProgress color="secondary" />
          </Box>
        )}
        <div ref={chatEndRef} />
      </Box>

      {/* ERROR ALERTS */}
      {error && <Alert severity="error" sx={{ mb: 2, borderRadius: '12px', flexShrink: 0 }}>{error}</Alert>}

      <Paper elevation={3} sx={{ ...cardStyle, flexShrink: 0, p: 2, display: 'flex', gap: 2, alignItems: 'center', width: '100%' }}>
        <TextField
          fullWidth
          multiline
          maxRows={3}
          variant="outlined"
          placeholder="Ask a question..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={loading}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          sx={{ flexGrow: 1 }}
        />
        <Button
          variant="contained"
          color="secondary"
          disabled={loading || !question.trim()}
          onClick={handleSubmit}
          sx={{ height: '56px', minWidth: '80px', borderRadius: '12px' }}
        >
          <Send />
        </Button>
      </Paper>

    </Container>
  );
};

export default QAPage;