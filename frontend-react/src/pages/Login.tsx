import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import Button from '../components/Button';
import { Activity } from 'lucide-react';

export default function Login() {
  const navigate = useNavigate();
  const { signIn, signUp } = useAuth();
  const { isDark } = useTheme();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      if (isSignUp) {
        await signUp(email, password);
        alert('Sign up successful! Please check your email to confirm.');
      } else {
        await signIn(email, password);
        navigate('/');
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center ${isDark ? 'bg-gray-900' : 'bg-gray-100'}`}>
      <div className={`w-full max-w-md ${isDark ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-lg p-8`}>
        <div className="flex items-center justify-center gap-2 mb-8">
          <div className={`w-10 h-10 rounded-lg ${isDark ? 'bg-blue-600' : 'bg-blue-500'}`}></div>
          <h1 className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
            Ki-67 Analyzer
          </h1>
        </div>

        <h2 className={`text-2xl font-bold mb-6 text-center ${isDark ? 'text-white' : 'text-gray-900'}`}>
          {isSignUp ? 'Create Account' : 'Welcome Back'}
        </h2>

        {error && (
          <div className={`p-4 rounded-lg mb-6 ${isDark ? 'bg-red-900 text-red-200' : 'bg-red-50 text-red-700'}`}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
              Email Address
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                isDark
                  ? 'bg-gray-700 border-gray-600 text-white'
                  : 'bg-white border-gray-300 text-gray-900'
              }`}
              placeholder="you@example.com"
            />
          </div>

          <div>
            <label className={`block text-sm font-medium mb-2 ${isDark ? 'text-gray-200' : 'text-gray-700'}`}>
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className={`w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                isDark
                  ? 'bg-gray-700 border-gray-600 text-white'
                  : 'bg-white border-gray-300 text-gray-900'
              }`}
              placeholder="••••••••"
            />
          </div>

          <Button
            type="submit"
            disabled={loading}
            className="w-full"
          >
            {loading ? 'Loading...' : isSignUp ? 'Sign Up' : 'Sign In'}
          </Button>
        </form>

        <div className={`mt-6 text-center ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
          {isSignUp ? (
            <>
              Already have an account?{' '}
              <button
                onClick={() => {
                  setIsSignUp(false);
                  setError('');
                }}
                className="text-blue-500 hover:text-blue-600 font-medium"
              >
                Sign In
              </button>
            </>
          ) : (
            <>
              Don't have an account?{' '}
              <button
                onClick={() => {
                  setIsSignUp(true);
                  setError('');
                }}
                className="text-blue-500 hover:text-blue-600 font-medium"
              >
                Sign Up
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
