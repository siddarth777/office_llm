import React, { useState, useRef } from 'react';
import { Send, Paperclip, LogOut, User, Moon, Sun, Trash2, Bot, Zap, Mic, MicOff } from 'lucide-react';
import ChatMessage from './ChatMessage';

//http://127.0.0.1:8000
const BACKEND_PATH='http://192.168.137.98:8000'

const ChatInterface = ({ user, onLogout, darkMode, toggleDarkMode }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: `Hello ${user?.name}! I'm V, an AI assistant. How can I help you today?`,
      sender: 'varuna',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [currentModel, setCurrentModel] = useState('model1'); // Default model
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  
  // Speech recording states
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);
  
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Model configurations
  const models = {
    model1: {
      name: 'Standard AI',
      id:'llama3.1:8b',
      icon: <Bot size={16} />,
      description: 'Balanced performance model'
    },
  };

  // Close dropdown when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (event) => {
      if (isDropdownOpen && !event.target.closest('.model-dropdown')) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isDropdownOpen]);

  // Initial greeting message
  const getInitialMessage = () => ({
    id: 1,
    text: `Hello ${user?.name}! I'm V, an AI assistant. How can I help you today?`,
    sender: 'varuna',
    timestamp: new Date()
  });

  // Function to convert messages array to string format
  const messagesToString = (messagesArray) => {
    return messagesArray.map(msg => {
      const role = msg.sender === 'user' ? 'User' : 'V';
      return `${role}: ${msg.text}`;
    }).join('\n');
  };

  // Speech recording functions
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        }
      });

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm;codecs=opus' });
        await sendAudioToBackend(audioBlob);
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start(1000); // Collect data every second
      setIsRecording(true);
    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Could not access microphone. Please check your permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessingAudio(true);
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('audio', audioBlob, 'recording.webm');

      const response = await fetch(BACKEND_PATH + '/speech-to-text', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Add transcribed text to input box
      if (data.text) {
        setInputText(prev => prev + (prev ? ' ' : '') + data.text);
      } else if (data.transcription) {
        setInputText(prev => prev + (prev ? ' ' : '') + data.transcription);
      } else {
        console.warn('No text received from speech-to-text service');
      }

    } catch (error) {
      console.error('Error processing audio:', error);
      alert('Failed to process audio. Please try again.');
    } finally {
      setIsProcessingAudio(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleModelSwitch = async (newModel) => {
    if (newModel === currentModel) return;

    setCurrentModel(newModel);
    
    // Add a system message to the chat indicating model switch
    const modelSwitchMessage = {
      id: Date.now(),
      text: `Switched to ${models[newModel].name}`,
      sender: 'system',
      timestamp: new Date(),
      isSystemMessage: true
    };

    setMessages(prev => [...prev, modelSwitchMessage]);

    try {
      // Notify backend about model switch
      const response = await fetch(BACKEND_PATH+'/switch-model', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelName: models[newModel].id
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Model switch response:', data);
      
    } catch (error) {
      console.error('Error switching model:', error);
      
      // Add error message if model switch fails
      const errorMessage = {
        id: Date.now() + 1,
        text: `Failed to switch to ${models[newModel].name}. Please try again.`,
        sender: 'system',
        timestamp: new Date(),
        isSystemMessage: true,
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;
    
    // Convert chat history to string
    const chatHistoryString = messagesToString(messages);
    
    const currentInput = inputText;
    setInputText('');
    setIsTyping(true);

    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    // Update messages with the new user message
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    try {
      const response = await fetch(BACKEND_PATH+'/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: currentInput,
          chatHistory: chatHistoryString,
          model: currentModel // Include current model in the request
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const apiResponse = {
        id: Date.now() + 1,
        text: data.response || data.message || 'No response received',
        sender: 'varuna',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, apiResponse]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorResponse = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error while processing your message. Please try again.',
        sender: 'varuna',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearChat = () => {
    setMessages([getInitialMessage()]);
    setInputText('');
    setIsTyping(false);
  };

  const handleFileAttach = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    console.log('File selected:', file.name);

    // Add a file message to the chat
    const fileMessage = {
      id: Date.now(),
      text: `📎 Uploaded file: ${file.name}`,
      sender: 'user',
      timestamp: new Date(),
      isFile: true
    };

    setMessages(prev => [...prev, fileMessage]);
    setIsTyping(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model', currentModel); // Include current model

      const response = await fetch(BACKEND_PATH+'/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const apiResponse = {
        id: Date.now() + 1,
        text: data.response || data.message || 'File uploaded successfully',
        sender: 'varuna',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, apiResponse]);
    } catch (error) {
      console.error('Error uploading file:', error);
      
      const errorResponse = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error while uploading the file. Please try again.',
        sender: 'varuna',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorResponse]);
    } finally {
      setIsTyping(false);
      // Clear the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  return (
    <div className="chat-container">
      <header className="chat-header">
        <div className="header-content">
          <div className="header-left">
            <h1>V Chat</h1>
            <div className="model-selector">
              <div className="model-dropdown">
                <button
                  className="model-dropdown-trigger"
                  onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                  type="button"
                >
                  <div className="current-model">
                    {models[currentModel].icon}
                    <span>{models[currentModel].name}</span>
                  </div>
                  <svg 
                    className={`dropdown-arrow ${isDropdownOpen ? 'open' : ''}`}
                    width="12" 
                    height="12" 
                    viewBox="0 0 12 12" 
                    fill="currentColor"
                  >
                    <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                  </svg>
                </button>
                
                {isDropdownOpen && (
                  <div className="model-dropdown-menu">
                    {Object.entries(models).map(([modelKey, modelInfo]) => (
                      <button
                        key={modelKey}
                        onClick={() => {
                          handleModelSwitch(modelKey);
                          setIsDropdownOpen(false);
                        }}
                        className={`model-dropdown-item ${currentModel === modelKey ? 'active' : ''}`}
                      >
                        <div className="model-info">
                          {modelInfo.icon}
                          <div className="model-details">
                            <span className="model-name">{modelInfo.name}</span>
                            <span className="model-description">{modelInfo.description}</span>
                          </div>
                        </div>
                        {currentModel === modelKey && (
                          <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor" className="check-icon">
                            <path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/>
                          </svg>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="user-info">
            <button 
              onClick={handleClearChat}
              className="clear-chat-button header-toggle"
              title="Clear chat history"
            >
              <Trash2 size={16} />
            </button>
            <button 
              onClick={toggleDarkMode}
              className="dark-mode-toggle header-toggle"
              title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {darkMode ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <span className="user-name">
              <User size={16} />
              {user?.name}
            </span>
            <button onClick={onLogout} className="logout-button">
              <LogOut size={16} />
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="chat-messages">
        {messages.map(message => (
          <ChatMessage key={message.id} message={message} />
        ))}
        {isTyping && (
          <div className="typing-indicator">
            <div className="typing-message">
              <div className="avatar varuna-avatar">V</div>
              <div className="typing-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>

      <form onSubmit={handleSendMessage} className="chat-input-form">
        <div className="input-container">
          <button
            type="button"
            onClick={handleFileAttach}
            className="attach-button"
            title="Attach file"
          >
            <Paperclip size={20} />
          </button>
          
          <button
            type="button"
            onClick={toggleRecording}
            className={`speech-button ${isRecording ? 'recording' : ''} ${isProcessingAudio ? 'processing' : ''}`}
            title={isRecording ? 'Stop recording' : isProcessingAudio ? 'Processing audio...' : 'Start voice recording'}
            disabled={isProcessingAudio}
          >
            {isProcessingAudio ? (
              <div className="processing-spinner">
                <div className="spinner"></div>
              </div>
            ) : 
              <Mic size={20}/>
            }
          </button>
          
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder={`Message V (${models[currentModel].name})...`}
            className="chat-input"
          />
          
          <button 
            type="submit" 
            className="send-button"
            disabled={!inputText.trim()}
            title="Send message"
          >
            <Send size={20} />
          </button>
        </div>
        
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          accept="*/*"
        />
      </form>
    </div>
  );
};

export default ChatInterface;