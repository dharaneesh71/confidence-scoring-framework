import React, { useState } from 'react';
import {
  Container, Paper, TextField, Button, Typography, Box,
  CircularProgress, Alert, Chip, Card, CardContent, LinearProgress,
  useTheme, Divider, Rating, Collapse
} from '@mui/material';
import { 
  Send, Lightbulb, CheckCircle, Public, WarningAmber
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext'; // <--- NEW IMPORT
import '../styles/QAPage.css';

const QAPage = () => {
  const theme = useTheme();
  const { user } = useAuth(); // <--- GET USER TOKEN
  
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Feedback State
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState('');
  const [feedbackSent, setFeedbackSent] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) { setError('Please enter a question'); return; }
    
    setLoading(true); 
    setError(null); 
    setResult(null);
    setFeedbackSent(false);
    setRating(0);
    setComment('');

    try {
      // Direct fetch to include Authorization Header
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}` // <--- SEND TOKEN
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) throw new Error("Failed to get answer");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    if (!result?.history_id) return;

    try {
      await fetch('http://localhost:8000/api/feedback', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${user.token}`
        },
        body: JSON.stringify({
          history_id: result.history_id,
          rating: rating,
          comment: comment
        }),
      });
      setFeedbackSent(true);
    } catch (err) {
      console.error("Feedback failed", err);
    }
  };

  const isGeneralKnowledge = (res) => res && res.confidence_score < 0.1;

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

  const cardStyle = {
    p: 4, 
    borderRadius: '16px',
    background: theme.palette.mode === 'dark' ? 'rgba(30, 41, 59, 0.8)' : '#ffffff',
    boxShadow: theme.palette.mode === 'dark' ? '0 8px 32px rgba(0,0,0,0.3)' : '0 8px 32px rgba(0,0,0,0.05)',
  };

  return (
    <Container maxWidth="lg" className="qa-container">
      <Box sx={{ my: 6, textAlign: 'center' }}>
        
        <Typography variant="h1" className="glassy-text" sx={{ mb: 1 }}>
          CONFID.AI
        </Typography>
        
        <Typography variant="subtitle1" sx={{ color: 'text.secondary', mb: 6, fontSize: '1.2rem', letterSpacing: '1px' }}>
          Multi-Dimensional Evidence-Based Evaluation
        </Typography>

        {/* INPUT SECTION */}
        <Paper elevation={3} sx={cardStyle}>
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              multiline
              rows={3}
              variant="outlined"
              label="Ask the AI..."
              placeholder="e.g., What are the properties of a square?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
              sx={{ mb: 3 }}
            />
            <Button
              type="submit"
              variant="contained"
              color="secondary"
              size="large"
              fullWidth
              disabled={loading || !question.trim()}
              startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <Send />}
              sx={{ py: 1.5, fontSize: '1.1rem', fontWeight: 'bold', color: 'white' }}
            >
              {loading ? 'Analyzing...' : 'Submit Question'}
            </Button>
          </form>
        </Paper>

        {error && <Alert severity="error" sx={{ mt: 2, borderRadius: '12px' }}>{error}</Alert>}

        {/* RESULTS SECTION */}
        {result && !loading && (
          <Paper className="result-paper-fade" sx={{ ...cardStyle, mt: 4, textAlign: 'left' }}>
              
             {/* ANSWER TEXT */}
             <Box sx={{ mb: 3 }}>
               <Typography variant="h6" color="secondary" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                 <Lightbulb fontSize="small" /> Answer:
               </Typography>
               <Typography paragraph sx={{ lineHeight: 1.8, fontSize: '1.05rem', pl: 1 }}>
                  {result.answer}
               </Typography>
             </Box>

             <Divider sx={{ my: 3 }} />
             
             {/* CONFIDENCE BLOCK */}
             <Box sx={{ my: 3, p: 3, border: `1px solid ${theme.palette.divider}`, borderRadius: '16px', bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)' }}>
                {isGeneralKnowledge(result) ? (
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
                          <Chip label={`${getConfidenceLabel(result.confidence_score)} CONFIDENCE`} color={getConfidenceColor(result.confidence_score)} sx={{ fontWeight: 'bold' }} />
                      </Box>
                      <LinearProgress variant="determinate" value={result.confidence_score * 100} color={getConfidenceColor(result.confidence_score)} sx={{ height: 10, borderRadius: 5, mb: 2 }} />
                      {result.explanation && (
                        <Box sx={{ p: 2, bgcolor: theme.palette.action.hover, borderRadius: '8px' }}>
                          <Typography variant="body2" color="text.secondary"><strong>Evaluation Details:</strong> {result.explanation}</Typography>
                        </Box>
                      )}
                   </>
                )}
             </Box>
             
             {/* CITATIONS */}
             {!isGeneralKnowledge(result) && result.citations && (
                 <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 2, fontWeight: 'bold', color: 'text.secondary' }}>SOURCE REFERENCES</Typography>
                    {result.citations.map((cit, idx) => (
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

             {/* NEW: FEEDBACK SECTION */}
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
        )}
      </Box>
    </Container>
  );
};

export default QAPage;