import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Box, Typography, List, ListItemButton, ListItemText, Drawer } from '@mui/material';

const Sidebar = ({ sessions, isOpen, onToggle }) => {
  const navigate = useNavigate();

  const groupSessions = (sessionsList) => {
    const groups = { Today: [], Yesterday: [], "Previous 7 Days": [] };
    
    // Normalize "now" to midnight local time
    const now = new Date();
    now.setHours(0, 0, 0, 0); 

    const sorted = [...sessionsList].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

    sorted.forEach(s => {
      // Force UTC parsing to fix timezone drift
      const dateString = s.created_at.endsWith('Z') ? s.created_at : `${s.created_at}Z`;
      const date = new Date(dateString);
      
      // Normalize chat date to midnight local time
      date.setHours(0, 0, 0, 0); 
      
      const diffDays = Math.round((now - date) / (1000 * 60 * 60 * 24));
      
      // If diffDays is 0 (or negative due to lingering drift), it's Today
      if (diffDays <= 0) groups.Today.push(s);
      else if (diffDays === 1) groups.Yesterday.push(s);
      else if (diffDays <= 7) groups["Previous 7 Days"].push(s);
    });
    
    return groups;
  };

  const groupedData = groupSessions(sessions);

  return (
    <Drawer variant="temporary" anchor="left" open={isOpen} onClose={onToggle}>
      <Box sx={{ width: 280, bgcolor: '#0f172a', color: 'white', height: '100%', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, borderBottom: '1px solid #1e293b' }}>
          <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#00d1ff' }}>CHAT HISTORY</Typography>
        </Box>
        
        <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 1 }}>
          {sessions.length === 0 ? (
            <Typography sx={{ p: 2, color: 'gray', textAlign: 'center' }}>No history yet.</Typography>
          ) : (
            Object.entries(groupedData).map(([label, items]) => (
              items.length > 0 && (
                <Box key={label} sx={{ mt: 1, mb: 2 }}>
                  <Typography variant="caption" sx={{ px: 2, color: '#64748b', fontWeight: 'bold', textTransform: 'uppercase' }}>
                    {label}
                  </Typography>
                  <List disablePadding>
                    {items.map((session) => (
                      <ListItemButton 
                        key={session.id} 
                        onClick={() => { 
                          navigate(`/chat/${session.id}`); 
                          onToggle(); // Close drawer after clicking a chat
                        }}
                        sx={{ borderRadius: 1, mx: 1, '&:hover': { bgcolor: 'rgba(0, 209, 255, 0.1)' } }}
                      >
                        <ListItemText 
                          primary={session.title || "New Chat"} 
                          primaryTypographyProps={{ noWrap: true, fontSize: '0.9rem' }} 
                        />
                      </ListItemButton>
                    ))}
                  </List>
                </Box>
              )
            ))
          )}
        </Box>
      </Box>
    </Drawer>
  );
};

export default Sidebar;