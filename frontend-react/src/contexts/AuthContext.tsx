import React, { createContext, useContext, useEffect, useState } from 'react';

interface User {
  id: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signUp: (email: string, password: string) => Promise<void>;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for stored user session (demo mode - auto-login)
    const storedUser = localStorage.getItem('ki67_user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    } else {
      // Auto-login for demo
      const demoUser = {
        id: 'demo-user-001',
        email: 'demo@ki67.local'
      };
      localStorage.setItem('ki67_user', JSON.stringify(demoUser));
      setUser(demoUser);
    }
    setLoading(false);
  }, []);

  const signUp = async (email: string, password: string) => {
    // Demo mode - accept any credentials
    const newUser = {
      id: `user-${Date.now()}`,
      email
    };
    localStorage.setItem('ki67_user', JSON.stringify(newUser));
    setUser(newUser);
  };

  const signIn = async (email: string, password: string) => {
    // Demo mode - accept any credentials
    const newUser = {
      id: `user-${Date.now()}`,
      email
    };
    localStorage.setItem('ki67_user', JSON.stringify(newUser));
    setUser(newUser);
  };

  const signOut = async () => {
    localStorage.removeItem('ki67_user');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, signUp, signIn, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
