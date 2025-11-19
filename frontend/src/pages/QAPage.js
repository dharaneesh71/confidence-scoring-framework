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

  // Logic to determine if it's a General Knowledge answer
  const isGeneralKnowledge = (res) => {
    if (!res) return false;
    // Check for low score indicating general knowledge
    return res.confidence_score < 0.1;
  };

  const getConfidenceColor = (score) => score >= 0.8 ? 'success' : score >= 0.5 ? 'warning' : 'error';
  
  const getConfidenceLabel = (score) => {
    if (score >= 0.8) return 'HIGH';
    if (score >= 0.5) return 'MEDIUM';
    return 'LOW';
  };

  // Styles
  const cardStyle = {
    p: 4, 
    borderRadius: '16px',
    background: theme.palette.mode === 'dark' ? 'rgba(15, 23, 42, 0.8)' : 'rgba(255, 255, 255, 0.9)',
    backdropFilter: 'blur(10px)',
    boxShadow: theme.palette.mode === 'light' ? '0 8px 32px rgba(0,0,0,0.1)' : undefined
  };

  return (
    <Container maxWidth="lg" className="qa-container">
      <Box sx={{ my: 4, textAlign: 'center' }}>
        <Typography variant="h3" gutterBottom sx={{ 
          fontWeight: 800, 
          color: 'text.primary', 
          textShadow: theme.palette.mode === 'dark' ? '0 0 20px rgba(77, 208, 225, 0.5)' : 'none' 
        }}>
          AI Confidence Scoring
        </Typography>
        <Typography variant="subtitle1" sx={{ color: 'text.secondary', mb: 4 }}>
          Multi-Dimensional Evidence-Based Evaluation
        </Typography>

        {/* INPUT SECTION */}
        <Paper elevation={10} sx={cardStyle}>
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
                        boxShadow: `0 0 15px 1px ${theme.palette.secondary.main}40`
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
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {/* RESULTS SECTION */}
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
             
             {/* STATUS BLOCK (Dynamic: Confidence OR General Knowledge) */}
             <Box sx={{ 
                 my: 3, 
                 p: 3, 
                 border: `2px solid ${theme.palette.divider}`, 
                 borderRadius: '12px',
                 bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)'
             }}>
                {isGeneralKnowledge(result) ? (
                   // --- CASE 1: GENERAL KNOWLEDGE (No Score) ---
                   <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, color: 'text.secondary' }}>
                       <Public sx={{ fontSize: 36, color: 'info.main' }} />
                       <Box>
                           <Typography variant="h6" sx={{ fontWeight: 'bold', color: 'info.main', mb: 0.5 }}>
                               General Knowledge Response
                           </Typography>
                           <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                               Answer not found in Ground Truth documents. Confidence score cannot be calculated.
                           </Typography>
                       </Box>
                   </Box>
                ) : (
                   // --- CASE 2: GROUND TRUTH (Show Score) ---
                   <>
                      {/* Confidence Header */}
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <CheckCircle fontSize="medium" color="success" />
                              <Typography variant="h6" color="text.primary" sx={{ fontWeight: 'bold' }}>
                                  Verified from Documents
                              </Typography>
                          </Box>
                          <Chip 
                              label={`${getConfidenceLabel(result.confidence_score)} CONFIDENCE`} 
                              color={getConfidenceColor(result.confidence_score)} 
                              size="medium"
                              sx={{ fontWeight: 'bold', fontSize: '0.9rem' }}
                          />
                      </Box>

                      {/* Confidence Score Bar */}
                      <Box sx={{ mb: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                          <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 'bold' }}>
                            Confidence Score
                          </Typography>
                          <Typography variant="body2" color="text.primary" sx={{ fontWeight: 'bold' }}>
                            {(result.confidence_score * 100).toFixed(0)}%
                          </Typography>
                        </Box>
                        <LinearProgress 
                            variant="determinate" 
                            value={result.confidence_score * 100} 
                            color={getConfidenceColor(result.confidence_score)} 
                            sx={{ 
                              height: 10, 
                              borderRadius: 5,
                              backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
                            }} 
                        />
                      </Box>

                      {/* Detailed Explanation */}
                      {result.explanation && (
                        <Box sx={{ 
                          mt: 2, 
                          p: 2, 
                          bgcolor: theme.palette.mode === 'dark' ? 'rgba(77, 208, 225, 0.05)' : 'rgba(63, 81, 181, 0.05)',
                          borderRadius: '8px',
                          borderLeft: `4px solid ${theme.palette.secondary.main}`
                        }}>
                          <Typography variant="body2" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            <Info fontSize="small" />
                            <strong>Evaluation Details:</strong>
                          </Typography>
                          <Typography variant="body2" color="text.primary" sx={{ lineHeight: 1.6 }}>
                            {result.explanation}
                          </Typography>
                        </Box>
                      )}

                      {/* Four Dimensions Breakdown (if available) */}
                      {result.score_breakdown && (
                        <Box sx={{ mt: 3 }}>
                          <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2, fontWeight: 'bold' }}>
                            SCORING DIMENSIONS
                          </Typography>
                          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 2 }}>
                            
                            {/* Factual Consistency */}
                            {result.score_breakdown.consistency !== undefined && (
                              <Tooltip title="Every claim is supported by evidence">
                                <Card variant="outlined" sx={{ bgcolor: 'transparent', borderColor: 'divider' }}>
                                  <CardContent sx={{ textAlign: 'center', py: 2 }}>
                                    <FactCheck color="primary" sx={{ fontSize: 32, mb: 1 }} />
                                    <Typography variant="caption" color="text.secondary" display="block">
                                      Factual Consistency
                                    </Typography>
                                    <Typography variant="h6" color="text.primary" sx={{ fontWeight: 'bold' }}>
                                      {(result.score_breakdown.consistency * 100).toFixed(0)}%
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Tooltip>
                            )}

                            {/* Semantic Alignment */}
                            {result.score_breakdown.semantic !== undefined && (
                              <Tooltip title="Meaning matches ground truth">
                                <Card variant="outlined" sx={{ bgcolor: 'transparent', borderColor: 'divider' }}>
                                  <CardContent sx={{ textAlign: 'center', py: 2 }}>
                                    <Verified color="secondary" sx={{ fontSize: 32, mb: 1 }} />
                                    <Typography variant="caption" color="text.secondary" display="block">
                                      Semantic Alignment
                                    </Typography>
                                    <Typography variant="h6" color="text.primary" sx={{ fontWeight: 'bold' }}>
                                      {(result.score_breakdown.semantic * 100).toFixed(0)}%
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Tooltip>
                            )}

                            {/* Completeness */}
                            {result.score_breakdown.completeness !== undefined && (
                              <Tooltip title="All essential parts addressed">
                                <Card variant="outlined" sx={{ bgcolor: 'transparent', borderColor: 'divider' }}>
                                  <CardContent sx={{ textAlign: 'center', py: 2 }}>
                                    <ChecklistRtl color="success" sx={{ fontSize: 32, mb: 1 }} />
                                    <Typography variant="caption" color="text.secondary" display="block">
                                      Completeness
                                    </Typography>
                                    <Typography variant="h6" color="text.primary" sx={{ fontWeight: 'bold' }}>
                                      {(result.score_breakdown.completeness * 100).toFixed(0)}%
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Tooltip>
                            )}

                            {/* Precision */}
                            {result.score_breakdown.precision !== undefined && (
                              <Tooltip title="No hallucinated content">
                                <Card variant="outlined" sx={{ bgcolor: 'transparent', borderColor: 'divider' }}>
                                  <CardContent sx={{ textAlign: 'center', py: 2 }}>
                                    <WarningAmber color="warning" sx={{ fontSize: 32, mb: 1 }} />
                                    <Typography variant="caption" color="text.secondary" display="block">
                                      Precision
                                    </Typography>
                                    <Typography variant="h6" color="text.primary" sx={{ fontWeight: 'bold' }}>
                                      {(result.score_breakdown.precision * 100).toFixed(0)}%
                                    </Typography>
                                  </CardContent>
                                </Card>
                              </Tooltip>
                            )}
                          </Box>
                        </Box>
                      )}
                   </>
                )}
             </Box>
             
             {/* CITATIONS (Only show if verified from documents) */}
             {!isGeneralKnowledge(result) && result.citations && result.citations.length > 0 && (
                 <Box sx={{ mt: 3 }}>
                    <Divider sx={{ mb: 2 }} />
                    <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 2, fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Info fontSize="small" />
                      SOURCE REFERENCES
                    </Typography>
                    {result.citations.map((cit, idx) => (
                        <Card key={idx} variant="outlined" sx={{ 
                          mb: 1.5, 
                          bgcolor: 'transparent', 
                          borderColor: 'divider',
                          '&:hover': {
                            borderColor: 'secondary.main',
                            transform: 'translateX(4px)',
                            transition: 'all 0.2s'
                          }
                        }}>
                            <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                                  <Typography variant="caption" color="secondary" sx={{ fontWeight: 'bold' }}>
                                    {cit.source}
                                    {cit.page && ` - Page ${cit.page}`}
                                  </Typography>
                                  {cit.relevance_score !== undefined && (
                                    <Chip 
                                      label={`${(cit.relevance_score * 100).toFixed(0)}% match`} 
                                      size="small" 
                                      color="primary" 
                                      variant="outlined"
                                    />
                                  )}
                                </Box>
                                <Typography variant="body2" sx={{ 
                                  fontStyle: 'italic', 
                                  color: 'text.secondary', 
                                  mt: 0.5,
                                  pl: 1,
                                  borderLeft: `2px solid ${theme.palette.divider}`
                                }}>
                                  "{cit.excerpt}"
                                </Typography>
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