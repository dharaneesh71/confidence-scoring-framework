import React, { useState } from 'react';
import {
  Container, Paper, TextField, Button, Typography, Box,
  CircularProgress, Alert, Chip, Card, CardContent, LinearProgress,
  useTheme, Divider, Tooltip
} from '@mui/material';
import { 
  Send, Info, WarningAmber, CheckCircle, Public, 
  FactCheck, Lightbulb, ChecklistRtl, Verified 
} from '@mui/icons-material';
// Ensure this path matches your file structure
import { submitQuery } from '../services/api'; 
import '../styles/QAPage.css';

const QAPage = () => {
  const theme = useTheme();
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) { setError('Please enter a question'); return; }
    setLoading(true); setError(null); setResult(null);
    try {
      const response = await submitQuery(question);
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isGeneralKnowledge = (res) => {
    if (!res) return false;
    return res.confidence_score < 0.1;
  };

  // --- UPDATED THRESHOLD LOGIC (75%) ---
  const getConfidenceColor = (score) => {
    if (score >= 0.75) return 'success'; // Green starts at 75%
    if (score >= 0.5) return 'warning';
    return 'error';
  };
  
  const getConfidenceLabel = (score) => {
    if (score >= 0.75) return 'HIGH';    // High starts at 75%
    if (score >= 0.5) return 'MEDIUM';
    return 'LOW';
  };

  // Standard card style (Clean, no glass effect on surrounding area)
  const cardStyle = {
    p: 4, 
    borderRadius: '16px',
    // Adaptive background for Light/Dark mode
    background: theme.palette.mode === 'dark' ? 'rgba(30, 41, 59, 0.8)' : '#ffffff',
    boxShadow: theme.palette.mode === 'dark' ? '0 8px 32px rgba(0,0,0,0.3)' : '0 8px 32px rgba(0,0,0,0.05)',
  };

  return (
    <Container maxWidth="lg" className="qa-container">
      <Box sx={{ my: 6, textAlign: 'center' }}>
        
        {/* --- 1. SPECIAL TEXT EFFECT ONLY ON THE TITLE --- */}
        <Typography variant="h1" className="glassy-text" sx={{ mb: 1 }}>
          CONFID.AI
        </Typography>
        
        <Typography variant="subtitle1" sx={{ color: 'text.secondary', mb: 6, fontSize: '1.2rem', letterSpacing: '1px' }}>
          Multi-Dimensional Evidence-Based Evaluation
        </Typography>

        {/* --- 2. INPUT SECTION (Clean Standard Style) --- */}
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
              InputLabelProps={{
                sx: { color: 'text.secondary', '&.Mui-focused': { color: 'secondary.main' } }
              }}
              sx={{ 
                mb: 3,
                '& .MuiOutlinedInput-root': {
                    backgroundColor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : '#f8f9fa',
                    '&:hover fieldset': { borderColor: 'secondary.main' },
                    '&.Mui-focused fieldset': { 
                        borderColor: 'secondary.main', 
                    },
                },
              }}
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

        {/* ERROR DISPLAY */}
        {error && (
          <Alert severity="error" sx={{ mt: 2, borderRadius: '12px' }}>
            {error}
          </Alert>
        )}

        {/* --- 3. RESULTS SECTION (Clean Standard Style) --- */}
        {result && !loading && (
          <Paper className="result-paper-fade" sx={{ ...cardStyle, mt: 4, textAlign: 'left' }}>
             
             {/* ANSWER TEXT */}
             <Box sx={{ mb: 3 }}>
               <Typography variant="h6" color="secondary" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                 <Lightbulb fontSize="small" />
                 Answer:
               </Typography>
               <Typography paragraph sx={{ 
                 lineHeight: 1.8, 
                 color: 'text.primary',
                 fontSize: '1.05rem',
                 pl: 1
               }}>
                  {result.answer}
               </Typography>
             </Box>

             <Divider sx={{ my: 3 }} />
             
             {/* STATUS BLOCK */}
             <Box sx={{ 
                 my: 3, 
                 p: 3, 
                 border: `1px solid ${theme.palette.divider}`, 
                 borderRadius: '16px',
                 bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)'
             }}>
                {isGeneralKnowledge(result) ? (
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
                              <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                                  Verified from Documents
                              </Typography>
                          </Box>
                          <Chip 
                              label={`${getConfidenceLabel(result.confidence_score)} CONFIDENCE`} 
                              color={getConfidenceColor(result.confidence_score)} 
                              sx={{ fontWeight: 'bold' }}
                          />
                      </Box>

                      <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="body2" color="text.secondary">Confidence Score</Typography>
                          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            {(result.confidence_score * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                        {/* Progress bar reflects the new color logic */}
                        <LinearProgress 
                            variant="determinate" 
                            value={result.confidence_score * 100} 
                            color={getConfidenceColor(result.confidence_score)} 
                            sx={{ height: 10, borderRadius: 5 }} 
                        />
                      </Box>

                      {result.explanation && (
                        <Box sx={{ mt: 2, p: 2, bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.04)', borderRadius: '8px' }}>
                          <Typography variant="body2" color="text.secondary">
                            <strong>Evaluation Details:</strong> {result.explanation}
                          </Typography>
                        </Box>
                      )}

                      {/* Score Breakdown Grid */}
                      {result.score_breakdown && (
                        <Box sx={{ mt: 3, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 2 }}>
                            {Object.entries(result.score_breakdown).map(([key, score]) => (
                                <Box key={key} sx={{ textAlign: 'center', p: 2, borderRadius: '12px', bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'white', border: `1px solid ${theme.palette.divider}` }}>
                                    <Typography variant="caption" sx={{ color: 'text.secondary', textTransform: 'capitalize' }}>{key}</Typography>
                                    <Typography variant="h6" sx={{ fontWeight: 'bold' }}>{(score * 100).toFixed(0)}%</Typography>
                                </Box>
                            ))}
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
          </Paper>
        )}
      </Box>
    </Container>
  );
};

export default QAPage;