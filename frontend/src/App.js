/**
 * Main App Component with Routing
 */
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Box,
  CssBaseline,
  ThemeProvider,
  createTheme,
} from '@mui/material';
import { QuestionAnswer, AdminPanelSettings } from '@mui/icons-material';
import QAPage from './pages/QAPage';
import AdminPage from './pages/AdminPage';
import './App.css';

// Create Material-UI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h3: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ flexGrow: 1, minHeight: '100vh', bgcolor: 'grey.50' }}>
          {/* Navigation Bar */}
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Confidence Scoring Framework
              </Typography>
              <Button
                color="inherit"
                component={Link}
                to="/"
                startIcon={<QuestionAnswer />}
              >
                Q&A
              </Button>
              <Button
                color="inherit"
                component={Link}
                to="/admin"
                startIcon={<AdminPanelSettings />}
              >
                Admin
              </Button>
            </Toolbar>
          </AppBar>

          {/* Routes */}
          <Routes>
            <Route path="/" element={<QAPage />} />
            <Route path="/admin" element={<AdminPage />} />
          </Routes>

          {/* Footer */}
          <Box
            component="footer"
            sx={{
              py: 3,
              px: 2,
              mt: 'auto',
              backgroundColor: 'grey.200',
            }}
          >
            <Container maxWidth="lg">
              <Typography variant="body2" color="text.secondary" align="center">
                © 2024 Confidence Scoring Framework | Team: Nipun, Jaideep, Dharaneesh
              </Typography>
            </Container>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;