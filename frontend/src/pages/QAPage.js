/**
 * Main Q&A Interface Page
 */
import React, { useState } from 'react';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  CircularProgress,
  Alert,
  Chip,
  Card,
  CardContent,
  Divider,
} from '@mui/material';
import { Send, Info } from '@mui/icons-material';
import { submitQuery } from '../services/api';
import '../styles/QAPage.css';

const QAPage = () => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await submitQuery(question);
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.5) return 'warning';
    return 'error';
  };

  const getConfidenceColorHex = (score) => {
    if (score >= 0.8) return '#4caf50';
    if (score >= 0.5) return '#ff9800';
    return '#f44336';
  };

  return (
    <Container maxWidth="lg" className="qa-container">
      <Box sx={{ my: 4 }}>
        {/* Header */}
        <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ mb: 1 }}>
          AI Confidence Scoring Framework
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
          Ask questions and receive AI-generated answers with confidence scores
        </Typography>

        {/* Question Input Form */}
        <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
          <form onSubmit={handleSubmit}>
            <TextField
              fullWidth
              multiline
              rows={3}
              variant="outlined"
              label="Enter your question"
              placeholder="e.g., What is the confidence scoring framework?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
              sx={{ mb: 2 }}
            />
            <Button
              type="submit"
              variant="contained"
              size="large"
              fullWidth
              disabled={loading || !question.trim()}
              startIcon={loading ? <CircularProgress size={20} /> : <Send />}
            >
              {loading ? 'Processing...' : 'Submit Question'}
            </Button>
          </form>
        </Paper>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Loading State */}
        {loading && (
          <Paper elevation={2} sx={{ p: 4, textAlign: 'center' }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              Processing your question...
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Generating answer, searching knowledge base, and computing confidence score
            </Typography>
          </Paper>
        )}

        {/* Results Display */}
        {result && !loading && (
          <Paper elevation={3} sx={{ p: 4 }}>
            {/* Question */}
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              Question:
            </Typography>
            <Typography variant="body1" paragraph sx={{ mb: 3 }}>
              {result.question}
            </Typography>

            {/* Answer */}
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              Answer:
            </Typography>
            <Typography variant="body1" paragraph sx={{ mb: 3, lineHeight: 1.8 }}>
              {result.answer}
            </Typography>

            <Divider sx={{ my: 3 }} />

            {/* Confidence Score */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                Confidence Score:
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                <Chip
                  label={`${(result.confidence_score * 100).toFixed(0)}%`}
                  color={getConfidenceColor(result.confidence_score)}
                  sx={{ fontSize: '1.2rem', padding: '24px 16px', fontWeight: 'bold' }}
                />
                <Chip
                  label={result.confidence_label}
                  variant="outlined"
                  color={getConfidenceColor(result.confidence_score)}
                  sx={{ fontSize: '1rem', padding: '20px 12px' }}
                />
                <Box sx={{ flex: 1, minWidth: '200px' }}>
                  <Box
                    sx={{
                      width: '100%',
                      height: '20px',
                      bgcolor: 'grey.200',
                      borderRadius: '10px',
                      overflow: 'hidden',
                    }}
                  >
                    <Box
                      sx={{
                        width: `${result.confidence_score * 100}%`,
                        height: '100%',
                        bgcolor: getConfidenceColorHex(result.confidence_score),
                        transition: 'width 0.5s ease-in-out',
                      }}
                    />
                  </Box>
                </Box>
              </Box>
              {result.processing_time_ms && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Processed in {result.processing_time_ms.toFixed(0)}ms
                </Typography>
              )}
            </Box>

            <Divider sx={{ my: 3 }} />

            {/* Citations */}
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main', display: 'flex', alignItems: 'center', gap: 1 }}>
              <Info fontSize="small" />
              Supporting Evidence:
            </Typography>
            {result.citations && result.citations.length > 0 ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {result.citations.map((citation, index) => (
                  <Card key={index} variant="outlined">
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 1 }}>
                        <Typography variant="subtitle2" color="primary">
                          Source: {citation.source}
                          {citation.page && ` (Page ${citation.page})`}
                        </Typography>
                        <Chip
                          label={`${(citation.similarity_score * 100).toFixed(0)}% match`}
                          size="small"
                          color={getConfidenceColor(citation.similarity_score)}
                          variant="outlined"
                        />
                      </Box>
                      <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                        "{citation.excerpt}"
                      </Typography>
                    </CardContent>
                  </Card>
                ))}
              </Box>
            ) : (
              <Alert severity="info">No citations available</Alert>
            )}
          </Paper>
        )}
      </Box>
    </Container>
  );
};

export default QAPage;