import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Button, Container, Box,
  CssBaseline, ThemeProvider, createTheme, IconButton,
} from '@mui/material';
import { QuestionAnswer, AdminPanelSettings, Brightness4, Brightness7 } from '@mui/icons-material';
import QAPage from './pages/QAPage';
import AdminPage from './pages/AdminPage';
import './App.css';

function App() {
  const [mode, setMode] = React.useState('dark'); // Default to dark

  const colorMode = React.useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
      },
    }),
    [],
  );

  const theme = React.useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: {
            main: mode === 'light' ? '#3F51B5' : '#7986CB',
          },
          secondary: {
            main: mode === 'light' ? '#008394' : '#4DD0E1', // Darker cyan for light mode visibility
          },
          background: {
            default: 'transparent', // Important: Let CSS background show
            paper: mode === 'light' ? '#FFFFFF' : '#0f172a',
          },
          text: {
            primary: mode === 'light' ? '#1a2027' : '#ffffff',
            secondary: mode === 'light' ? '#5f748d' : '#94a3b8',
          }
        },
        typography: {
          fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
          h3: { fontWeight: 700 },
        },
        components: {
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundImage: 'none',
                borderRadius: '16px',
              },
            },
          },
        },
      }),
    [mode],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {/* Apply CSS class dynamically here */}
      <div className={`App ${mode === 'dark' ? 'theme-dark' : 'theme-light'}`}>
        <Router>
          <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
            <AppBar 
              position="static" 
              color="transparent" 
              elevation={0} 
              sx={{ 
                backdropFilter: 'blur(10px)',
                borderBottom: mode === 'dark' ? '1px solid rgba(255,255,255,0.1)' : '1px solid rgba(0,0,0,0.1)',
                bgcolor: mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.5)'
              }}
            >
              <Toolbar>
                <Typography variant="h6" component="div" sx={{ flexGrow: 1, fontWeight: 'bold', letterSpacing: 1, color: 'text.primary' }}>
                  CONFIDENCE SCORING AI
                </Typography>
                <IconButton sx={{ ml: 1 }} onClick={colorMode.toggleColorMode} color="inherit">
                  {theme.palette.mode === 'dark' ? <Brightness7 /> : <Brightness4 sx={{ color: 'text.primary' }} />}
                </IconButton>
                <Button component={Link} to="/" startIcon={<QuestionAnswer />} sx={{ color: 'text.primary' }}>Q&A</Button>
                <Button component={Link} to="/admin" startIcon={<AdminPanelSettings />} sx={{ color: 'text.primary' }}>Admin</Button>
              </Toolbar>
            </AppBar>

            <Routes>
              <Route path="/" element={<QAPage />} />
              <Route path="/admin" element={<AdminPage />} />
            </Routes>

            <Box component="footer" sx={{ py: 3, px: 2, mt: 'auto', backdropFilter: 'blur(5px)', borderTop: '1px solid divider' }}>
              <Container maxWidth="lg">
                <Typography variant="body2" color="text.secondary" align="center">
                  © 2024 Confidence Scoring Framework
                </Typography>
              </Container>
            </Box>
          </Box>
        </Router>
      </div>
    </ThemeProvider>
  );
}

export default App;