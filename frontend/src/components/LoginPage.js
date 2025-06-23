import React, { useState } from 'react';
import { User, Lock, Moon, Sun } from 'lucide-react';

const BACKEND_PATH='http://192.168.137.98:8000'

const LoginPage = ({ onLogin, darkMode, toggleDarkMode }) => {
  const [formData, setFormData] = useState({
    name: '',
    password: ''
  });
  const [errors, setErrors] = useState({});
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.password.trim()) {
      newErrors.password = 'Password is required';
    }
    
    return newErrors;
  };

  const handleSubmit = async (e) => {
  e.preventDefault();
  const newErrors = validateForm();
  if (Object.keys(newErrors).length > 0) {
    setErrors(newErrors);
    return;
  }
  setIsLoading(true);
  try {
    const response = await fetch(BACKEND_PATH+'/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: formData.name,
        password: formData.password
      }),
    });
    const data = await response.json();
    if (response.ok) {
      console.log('Login successful:', data);
      onLogin({
        name: data.username,
        message: data.message
      });
    } else {
      // Check if status code is 401 and show popup
      if (response.status === 401) {
        alert('Login failed');
      }
      
      // Handle login errors
      setErrors({
        general: data.detail || 'Login failed. Please check your credentials.'
      });
    }
  } catch (error) {
    console.error('Login error:', error);
    setErrors({
      general: 'Network error. Please try again later.'
    });
  } finally {
    setIsLoading(false);
  }
};

  return (
    <div className="login-container">
      <button 
        onClick={toggleDarkMode}
        className="dark-mode-toggle"
        title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {darkMode ? <Sun size={20} /> : <Moon size={20} />}
      </button>
      
      <div className="login-card">
        <div className="login-header">
          <h1>Welcome to V Chat</h1>
          <p>Please enter your details to continue</p>
        </div>
        
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="name">
              <User size={18} />
              User Name
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleInputChange}
              className={errors.name ? 'error' : ''}
              placeholder="Enter your full name"
            />
            {errors.name && <span className="error-message">{errors.name}</span>}
          </div>

          <div className="form-group">
            <label htmlFor="password">
              <Lock size={18} />
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              className={errors.password ? 'error' : ''}
              placeholder="Enter your password"
            />
            {errors.password && <span className="error-message">{errors.password}</span>}
          </div>

          <button 
            type="submit" 
            className="login-button"
            disabled={isLoading}
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;