import os
import time
import signal
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse,StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mimetypes
import tempfile
from typing import Dict, Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

from sqlalchemy.ext.asyncio import AsyncSession
from contextlib import asynccontextmanager

from database.database import engine, AsyncSessionLocal
from database.models import Base, User
from database.crud import create_user, authenticate_user
import asyncio

import httpx
import subprocess
import json

from faster_whisper import WhisperModel


"python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m --host 0.0.0.0 --port 8080 --served-model-name facebook/opt-125m --disable-log-requests --tensor-parallel-size 1 --dtype auto --max-model-len 4096"


current_model = 'llama3.1:8b'
ollama_process = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Create database tables at startup
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"Database initialization error: {e}")
       
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "FastAPI Login Server with RAG is running"}

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    message: str
    username: str

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user with username and password
    """
    username = request.username.strip()
    password = request.password

    async with AsyncSessionLocal() as db:
        user = await authenticate_user(db, username, password)
       
        # Check if username exists
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
       
        return LoginResponse(
            message="Login successful",
            username=username
        )

class SwitchModelRequest(BaseModel):
    modelName: str

@app.post("/switch-model")
async def switch_model(request: SwitchModelRequest):
    global current_model

    model_id = request.modelName.strip()
   
    if not model_id:
        return JSONResponse({
            "status": "error",
            "message": "Model name cannot be empty"
        }, status_code=400)

    try:
        # Check if Ollama is running
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                await client.get("http://localhost:11434/api/tags")
            except Exception:
                return JSONResponse({
                    "status": "error",
                    "message": "Ollama server is not running"
                }, status_code=503)
                
        # Update current model
        current_model = model_id

        return JSONResponse({
            "status": "success",
            "message": f"Successfully switched to model: {model_id}",
            "current_model": current_model
        })

    except Exception as e:
        print(f"Error switching model: {e}")
        return JSONResponse({
            "status": "error",
            "message": f"Failed to switch model: {str(e)}"
        }, status_code=500)

class MessageRequest(BaseModel):
    chatHistory: str
    message: str

class MessageResponse(BaseModel):
    message: str

@app.post("/message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    """
    Send a message to the current llama.cpp model
    """
    history = request.chatHistory
    prompt = request.message
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    print(f"Current model: {current_model}")
    
    # Generating Chat History
    messages = []
    
    # Add system messages
    messages.append({
        "role": "system",
        "content": "You are a helpful AI assistant 'V'. Provide accurate, concise, and engaging responses. Be conversational and friendly while staying professional. Give direct answers with relevant context. Acknowledge uncertainty rather than guessing. Ask clarifying questions when needed."
    })
    
    if history and history.strip():
        #messages seperate by \n
        lines = history.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('V: '):
                messages.append({
                    "role": "assistant",
                    "content": line[3:]  # Remove 'V: '
                })
            elif line.startswith('User: '):
                messages.append({
                    "role": "user",
                    "content": line[6:]  # Remove 'User: '
                })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    print(f"Messages to send: {messages}")
    
    try:
        # Check if llama.cpp server is running
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Test connection first
            try:
                await client.get("http://localhost:8080/health", timeout=5.0)
            except Exception:
                # Fallback health check if /health doesn't exist
                try:
                    await client.get("http://localhost:8080/v1/models", timeout=5.0)
                except Exception:
                    raise HTTPException(status_code=503, detail="llama.cpp server is not responding")
            
            # Open API standard call
            print(f"Sending request to llama.cpp server")
            response = await client.post(
                url="http://localhost:8080/v1/chat/completions",
                json={
                    "model": current_model,
                    "messages": messages,
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "max_tokens": 1000,
                    "stream": False,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"]
                },
                headers={
                    "Content-Type": "application/json"
                },
                timeout=120.0
            )
            
            if response.status_code != 200:
                print(f"llama.cpp API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"llama.cpp API returned status {response.status_code}"
                )
            
            data = response.json()
            
            # Parse OpenAI-style response
            if "choices" not in data or not data["choices"]:
                print(f"Unexpected response format: {data}")
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response format from llama.cpp server"
                )
            
            ai_response = data["choices"][0]["message"]["content"].strip()
            
            if not ai_response:
                ai_response = "I apologize, but I couldn't generate a response. Please try again."
            
            print(f"Generated response:\n{ai_response}")
            return MessageResponse(message=ai_response)
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"llama.cpp Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/message/stream")
async def send_message_stream(request: MessageRequest):
    """
    Send a message to the current llama.cpp model with streaming response
    """
    history = request.chatHistory
    prompt = request.message
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    print(f"Current model: {current_model}")
    
    # Generating Chat History
    messages = []
    
    # Add system messages
    messages.append({
        "role": "system",
        "content": "You are a helpful AI assistant 'V'. Provide accurate, concise, and engaging responses. Be conversational and friendly while staying professional. Give direct answers with relevant context. Acknowledge uncertainty rather than guessing. Ask clarifying questions when needed."
    })
    
    if history and history.strip():
        #messages seperate by \n
        lines = history.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('V: '):
                messages.append({
                    "role": "assistant",
                    "content": line[3:]  # Remove 'V: '
                })
            elif line.startswith('User: '):
                messages.append({
                    "role": "user",
                    "content": line[6:]  # Remove 'User: '
                })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    print(f"Messages to send: {messages}")
    
    async def generate_stream():
        try:
            # Check if llama.cpp server is running
            async with httpx.AsyncClient(timeout=180.0) as client:
                # Test connection first
                try:
                    await client.get("http://localhost:8080/health", timeout=5.0)
                except Exception:
                    # Fallback health check if /health doesn't exist
                    try:
                        await client.get("http://localhost:8080/v1/models", timeout=5.0)
                    except Exception:
                        yield f"data: {json.dumps({'error': 'llama.cpp server is not responding'})}\n\n"
                        return
                
                # Open API standard call with streaming
                print(f"Sending streaming request to llama.cpp server")
                
                async with client.stream(
                    method="POST",
                    url="http://localhost:8080/v1/chat/completions",
                    json={
                        "model": current_model,
                        "messages": messages,
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "max_tokens": 1000,
                        "stream": True,  # Enable streaming
                        "stop": ["<|eot_id|>", "<|end_of_text|>"]
                    },
                    headers={
                        "Content-Type": "application/json"
                    },
                    timeout=180.0
                ) as response:
                    
                    if response.status_code != 200:
                        print(f"llama.cpp API error: {response.status_code}")
                        yield f"data: {json.dumps({'error': f'llama.cpp API returned status {response.status_code}'})}\n\n"
                        return
                    
                    # Process streaming response
                    async for chunk in response.aiter_lines():
                        if chunk:
                            # Remove 'data: ' prefix if present
                            if chunk.startswith('data: '):
                                chunk = chunk[6:]
                            
                            # Skip empty lines and [DONE] marker
                            if not chunk.strip() or chunk.strip() == '[DONE]':
                                continue
                            
                            try:
                                # Parse the JSON chunk
                                data = json.loads(chunk)
                                
                                # Extract the content from OpenAI-style streaming response
                                if "choices" in data and data["choices"]:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            # Send the token to the frontend
                                            yield f"data: {json.dumps({'token': content})}\n\n"
                                    
                                    # Check if streaming is finished
                                    if choice.get("finish_reason"):
                                        yield f"data: {json.dumps({'done': True})}\n\n"
                                        break
                                        
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}, chunk: {chunk}")
                                continue
                            except Exception as e:
                                print(f"Error processing chunk: {e}")
                                continue
                
        except HTTPException:
            yield f"data: {json.dumps({'error': 'HTTP exception occurred'})}\n\n"
        except Exception as e:
            print(f"llama.cpp Streaming Error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Error generating response: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )

@app.get("/models")
async def get_available_models():
    """
    Get list of available Ollama models
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return JSONResponse({
                    "status": "success",
                    "models": models,
                    "current_model": current_model
                })
            else:
                return JSONResponse({
                    "status": "error",
                    "message": "Failed to fetch models"
                }, status_code=500)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Error fetching models: {str(e)}"
        }, status_code=500)

@app.get("/health")
async def health_check():
    """
    Check if Ollama server is running and current model is available
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Ollama is responding
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                return JSONResponse({
                    "status": "unhealthy",
                    "message": "Ollama server not responding"
                }, status_code=503)
           
            # Check if current model is available
            data = response.json()
            available_models = [model["name"] for model in data.get("models", [])]
           
            return JSONResponse({
                "status": "healthy",
                "ollama_running": True,
                "current_model": current_model,
                "model_available": current_model in available_models,
                "available_models": available_models
            })
           
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "ollama_running": False
        }, status_code=503)

class FileUploadResponse(BaseModel):
    response: str
    file_info: dict
    status: str

# File type detection function
def get_file_type_info(file_path: str, filename: str) -> dict:
    """
    Get comprehensive file type information
    """
    file_info = {}
   
    # Get file extension
    _, extension = os.path.splitext(filename)
    file_info['extension'] = extension.lower() if extension else 'No extension'
   
    # Get MIME type using mimetypes module
    mime_type, _ = mimetypes.guess_type(filename)
    file_info['mime_type'] = mime_type or 'Unknown'
   
    # Get file size
    file_info['size_bytes'] = os.path.getsize(file_path)
    file_info['size_mb'] = round(file_info['size_bytes'] / (1024 * 1024), 2)
   
    return file_info

@app.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and analyze a file, then delete it
    """
    try:
        # Check if file is present
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
       
        # Create temporary file path
        temp_dir = tempfile.gettempdir()
        # Use original filename but make it safe
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        file_path = os.path.join(temp_dir, f"temp_{safe_filename}")
       
        try:
            # Save the file temporarily
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
           
            # Get file type information
            file_info = get_file_type_info(file_path, file.filename)
           
            # Create response message with file details
            response_message = f"""File Analysis Complete! 📁 File Details\\ Name: {file.filename}\\ Extension: {file_info['extension']}\\ Size:{file_info['size_mb']} MB ({file_info['size_bytes']:,} bytes)"""
           
            # Log file information (optional)
            print(f"File processed: {file.filename}")
            print(f"Extension: {file_info['extension']}")
            print(f"MIME Type: {file_info['mime_type']}")
            print(f"Size: {file_info['size_mb']} MB")
           
            return FileUploadResponse(
                response=response_message,
                file_info=file_info,
                status="success"
            )
           
        finally:
            # Always delete the file after processing
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File deleted: {file_path}")
   
    except Exception as e:
        # Clean up file if it exists and there was an error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
       
        print(f"Error processing file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the file: {str(e)}"
        )

# RAG-specific endpoints

class RAGUploadResponse(BaseModel):
    status: str
    message: str
    chunks_created: Optional[int] = None
    processing_time: Optional[float] = None
    filename: str

@app.post("/rag-upload", response_model=RAGUploadResponse)
async def rag_upload(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a document for RAG processing for a specific user
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
   
    # Validate user_id
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="User ID cannot be empty")
   
    user_id = user_id.strip()
   
    # Check supported file types
    file_extension = os.path.splitext(file.filename)[1].lower()
    supported_extensions = ['.pdf', '.txt', '.docx', '.doc']
   
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {', '.join(supported_extensions)}"
        )
   
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
    file_path = os.path.join(temp_dir, f"rag_{user_id}_{safe_filename}")
   
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
       
        # Process document for user
        result = await process_document_for_user(user_id, file_path, file.filename)
       
        if result["status"] == "success":
            return RAGUploadResponse(
                status="success",
                message=result["message"],
                chunks_created=result["chunks_created"],
                processing_time=result["processing_time"],
                filename=file.filename
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
   
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in RAG upload: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Temporary file deleted: {file_path}")

class RAGQueryRequest(BaseModel):
    user_id: str
    prompt: str
    max_results: Optional[int] = 4

class RAGQueryResponse(BaseModel):
    response: str
    sources: List[str]
    status: str

@app.post("/rag-query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system for a specific user
    """
    user_id = request.user_id.strip()
    prompt = request.prompt.strip()
   
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")
   
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
   
    try:
        # Get user's vector store
        vectorstore = await get_cached_vectorstore(user_id)
       
        if not vectorstore:
            raise HTTPException(
                status_code=404,
                detail=f"No documents found for user {user_id}. Please upload documents first."
            )
       
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": request.max_results}
        )
       
        # Create Ollama LLM
        llm = Ollama(
            base_url="http://localhost:11434",
            model=current_model,
            temperature=0.3
        )
       
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
       
        # Execute query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
       
        def run_query():
            return qa_chain({"query": prompt})
       
        result = await loop.run_in_executor(thread_pool, run_query)
       
        # Extract sources
        sources = []
        if "source_documents" in result:
            for doc in result["source_documents"]:
                source = doc.metadata.get("source", "Unknown")
                if source not in sources:
                    sources.append(source)
       
        return RAGQueryResponse(
            response=result["result"],
            sources=sources,
            status="success"
        )
       
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in RAG query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/rag-status/{user_id}")
async def rag_status(user_id: str):
    """
    Get RAG status for a specific user
    """
    user_id = user_id.strip()
   
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID cannot be empty")
   
    try:
        vectorstore_path = get_user_vectorstore_path(user_id)
        has_documents = vectorstore_path.exists()
       
        document_count = 0
        if has_documents:
            try:
                vectorstore = await get_cached_vectorstore(user_id)
                if vectorstore:
                    document_count = vectorstore.index.ntotal
            except Exception as e:
                print(f"Error getting document count: {e}")
       
        return JSONResponse({
            "user_id": user_id,
            "has_documents": has_documents,
            "document_count": document_count,
            "vectorstore_path": str(vectorstore_path),
            "embedding_model": EMBEDDING_MODEL,
            "status": "success"
        })
       
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Error getting RAG status: {str(e)}"
        }, status_code=500)
   
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    temp_webm_path = None
    temp_wav_path = None
   
    try:
        # Save uploaded WebM file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_webm_path = temp_file.name
       
        # Convert WebM to WAV using ffmpeg
        temp_wav_path = tempfile.mktemp(suffix=".wav")
       
        # Convert to WAV format (faster-whisper is more flexible with audio formats)
        subprocess.run([
            "ffmpeg", "-i", temp_webm_path,
            "-ar", "16000", "-ac", "1", "-f", "wav",
            temp_wav_path
        ], check=True, capture_output=True, stderr=subprocess.PIPE)
       
        # Transcribe audio with faster-whisper
        segments, info = whisper_model.transcribe(
            temp_wav_path,
            beam_size=5,  # Good balance between speed and accuracy
            language=None,  # Auto-detect language
            task="transcribe"  # Use "translate" if you want translation to English
        )
       
        # Extract text from segments
        transcribed_text = ""
        confidence_scores = []
       
        for segment in segments:
            transcribed_text += segment.text + " "
            if hasattr(segment, 'avg_logprob'):
                confidence_scores.append(segment.avg_logprob)
       
        # Clean up the text
        transcribed_text = transcribed_text.strip()
       
        # Calculate average confidence if available
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else None
       
        return JSONResponse(content={
            "text": transcribed_text,
            "status": "success"
        })
       
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        return JSONResponse(
            content={
                "error": f"Audio conversion failed: {error_message}",
                "status": "error"
            },
            status_code=500
        )
    except Exception as e:
        return JSONResponse(
            content={
                "error": f"Failed to process audio: {str(e)}",
                "status": "error"
            },
            status_code=500
        )
    finally:
        # Clean up temporary files
        if temp_webm_path and os.path.exists(temp_webm_path):
            os.unlink(temp_webm_path)
        if temp_wav_path and os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
          
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
export LD_LIBRARY_PATH=build/ggml/src:build/src/:$LD_LIBRARY_PATH

./build/bin/llama-server -m /home/caio/Downloads/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -c 2048 --port 8080 --host 0.0.0.0 --chat-template llama3
"""
