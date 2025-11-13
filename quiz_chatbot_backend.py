from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
import json
import os
import hashlib
from datetime import datetime
import pdfplumber
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import openai
import random
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# Database setup
def init_db():
    conn = sqlite3.connect('quiz_results.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quiz_sessions (
        session_id TEXT PRIMARY KEY,
        pdf_name TEXT,
        pdf_hash TEXT,
        upload_time TIMESTAMP,
        theme_data TEXT,
        quiz_data TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_name TEXT,
        score INTEGER,
        total_questions INTEGER,
        completion_time TIMESTAMP,
        detailed_results TEXT,
        FOREIGN KEY (session_id) REFERENCES quiz_sessions(session_id)
    )
    ''')
    conn.commit()
    conn.close()

init_db()

class PDFProcessor:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.index = None
        
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF with enhanced processing"""
        text = ""
        with pdfplumber.open(file_content) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Create overlapping chunks for better context preservation"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                # Add overlap
                overlap_sentences = current_chunk.split('. ')[-2:]
                current_chunk = '. '.join(overlap_sentences) + ". " + sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        self.chunks = chunks
        return chunks
    
    def create_embeddings(self, chunks: List[str]):
        """Create embeddings for chunks"""
        self.embeddings = embedding_model.encode(chunks)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[str]:
        """Find most similar chunks to a query"""
        query_embedding = embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.chunks[idx] for idx in indices[0]]

class ThemeDetector:
    def __init__(self):
        self.color_schemes = {
            "technology": {
                "primary": "#2563eb",
                "secondary": "#3b82f6", 
                "accent": "#60a5fa",
                "background": "#0f172a",
                "surface": "#1e293b",
                "text": "#f1f5f9"
            },
            "medical": {
                "primary": "#059669",
                "secondary": "#10b981",
                "accent": "#34d399", 
                "background": "#f0fdf4",
                "surface": "#dcfce7",
                "text": "#064e3b"
            },
            "business": {
                "primary": "#7c3aed",
                "secondary": "#8b5cf6",
                "accent": "#a78bfa",
                "background": "#faf5ff",
                "surface": "#ede9fe", 
                "text": "#2e1065"
            },
            "education": {
                "primary": "#dc2626",
                "secondary": "#ef4444",
                "accent": "#f87171",
                "background": "#fef2f2",
                "surface": "#fee2e2",
                "text": "#450a0a"
            },
            "science": {
                "primary": "#0891b2",
                "secondary": "#06b6d4",
                "accent": "#22d3ee",
                "background": "#f0fdfa",
                "surface": "#ccfbf1",
                "text": "#083344"
            },
            "default": {
                "primary": "#6366f1",
                "secondary": "#818cf8",
                "accent": "#a5b4fc",
                "background": "#f8fafc",
                "surface": "#e2e8f0",
                "text": "#1e293b"
            }
        }
        
        self.domain_keywords = {
            "technology": ["software", "programming", "code", "data", "algorithm", "computer", "digital", "api", "framework", "development"],
            "medical": ["health", "patient", "medical", "treatment", "disease", "clinical", "therapy", "diagnosis", "symptoms", "medicine"],
            "business": ["management", "strategy", "market", "finance", "revenue", "customer", "sales", "profit", "investment", "enterprise"],
            "education": ["learning", "student", "teacher", "curriculum", "education", "course", "academic", "university", "knowledge", "study"],
            "science": ["research", "experiment", "hypothesis", "scientific", "theory", "analysis", "observation", "data", "methodology", "results"]
        }
    
    def detect_domain(self, text: str) -> str:
        """Detect the domain of the PDF content"""
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for word in words if word in keywords)
            domain_scores[domain] = score
        
        # Get domain with highest score
        detected_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[detected_domain] < 5:  # Threshold for minimum matches
            detected_domain = "default"
        
        return detected_domain
    
    def get_theme_data(self, text: str, pdf_name: str) -> Dict:
        """Get comprehensive theme data including colors, icons, and styling"""
        domain = self.detect_domain(text)
        colors = self.color_schemes[domain]
        
        # Extract key topics for display
        doc = nlp(text[:5000])  # Process first 5000 chars for efficiency
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "PRODUCT", "EVENT"]]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
        
        # Get most common meaningful words
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]
        word_freq = Counter(words).most_common(10)
        
        return {
            "domain": domain,
            "colors": colors,
            "pdf_name": pdf_name,
            "key_topics": list(set(noun_phrases[:10])),
            "entities": list(set(entities[:10])),
            "top_words": [word for word, _ in word_freq],
            "icon_theme": self._get_icon_theme(domain),
            "gradient_style": self._get_gradient_style(domain)
        }
    
    def _get_icon_theme(self, domain: str) -> Dict:
        """Get icon suggestions based on domain"""
        icon_themes = {
            "technology": {"main": "💻", "secondary": "🚀", "accent": "⚡"},
            "medical": {"main": "🏥", "secondary": "💊", "accent": "🩺"},
            "business": {"main": "💼", "secondary": "📊", "accent": "💰"},
            "education": {"main": "📚", "secondary": "🎓", "accent": "✏️"},
            "science": {"main": "🔬", "secondary": "🧪", "accent": "🔭"},
            "default": {"main": "📄", "secondary": "❓", "accent": "✨"}
        }
        return icon_themes.get(domain, icon_themes["default"])
    
    def _get_gradient_style(self, domain: str) -> str:
        """Get gradient CSS based on domain"""
        gradients = {
            "technology": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "medical": "linear-gradient(135deg, #96e6a1 0%, #4bc0c8 100%)",
            "business": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
            "education": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            "science": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
            "default": "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)"
        }
        return gradients.get(domain, gradients["default"])

class QuizGenerator:
    def __init__(self, pdf_processor: PDFProcessor):
        self.pdf_processor = pdf_processor
        
    def generate_questions(self, text: str, num_questions: int = 10) -> List[Dict]:
        """Generate highly relevant questions based on PDF content"""
        questions = []
        
        # Process text with spaCy for better understanding
        doc = nlp(text[:10000])  # Limit for performance
        
        # Extract important sentences (those with entities or important noun phrases)
        important_sentences = []
        for sent in doc.sents:
            if any([ent for ent in sent.ents]) or len([chunk for chunk in sent.noun_chunks]) > 2:
                important_sentences.append(sent.text)
        
        # Generate different types of questions
        question_types = [
            self._generate_factual_question,
            self._generate_definition_question,
            self._generate_relationship_question,
            self._generate_true_false_question
        ]
        
        for i in range(min(num_questions, len(important_sentences))):
            sentence = important_sentences[i % len(important_sentences)]
            question_type = question_types[i % len(question_types)]
            
            try:
                question = question_type(sentence, doc)
                if question and self._validate_question(question, text):
                    questions.append(question)
            except:
                continue
        
        # Ensure we have enough questions
        while len(questions) < num_questions:
            # Generate from random chunks
            chunk = random.choice(self.pdf_processor.chunks)
            question = self._generate_question_from_chunk(chunk)
            if question and self._validate_question(question, text):
                questions.append(question)
        
        return questions[:num_questions]
    
    def _generate_factual_question(self, sentence: str, doc) -> Optional[Dict]:
        """Generate a factual question from a sentence"""
        sent_doc = nlp(sentence)
        entities = [ent for ent in sent_doc.ents]
        
        if not entities:
            return None
        
        # Pick an entity to ask about
        target_entity = random.choice(entities)
        
        # Create question by replacing entity with question word
        question_text = sentence.replace(target_entity.text, f"[{target_entity.label_}]")
        
        if target_entity.label_ in ["PERSON", "ORG"]:
            question_text = question_text.replace(f"[{target_entity.label_}]", "Who")
        elif target_entity.label_ in ["DATE", "TIME"]:
            question_text = question_text.replace(f"[{target_entity.label_}]", "When")
        elif target_entity.label_ in ["LOC", "GPE"]:
            question_text = question_text.replace(f"[{target_entity.label_}]", "Where")
        else:
            question_text = question_text.replace(f"[{target_entity.label_}]", "What")
        
        # Generate wrong options
        wrong_options = self._generate_wrong_options(target_entity.text, target_entity.label_, doc)
        
        if len(wrong_options) < 3:
            return None
        
        options = [target_entity.text] + wrong_options[:3]
        random.shuffle(options)
        
        return {
            "question": question_text + "?",
            "options": options,
            "correct_answer": target_entity.text,
            "explanation": f"The correct answer is found in the text: '{sentence}'",
            "type": "factual"
        }
    
    def _generate_definition_question(self, sentence: str, doc) -> Optional[Dict]:
        """Generate a definition-based question"""
        # Look for sentences with "is", "means", "defined as"
        definition_patterns = ["is", "means", "defined as", "refers to"]
        
        for pattern in definition_patterns:
            if pattern in sentence.lower():
                parts = sentence.lower().split(pattern)
                if len(parts) == 2:
                    term = parts[0].strip()
                    definition = parts[1].strip()
                    
                    if len(term.split()) <= 5:  # Reasonable term length
                        question_text = f"What {pattern} {term}?"
                        
                        # Generate plausible wrong definitions
                        wrong_options = [
                            definition.replace(random.choice(definition.split()), "different"),
                            "A completely unrelated concept",
                            "The opposite of " + term
                        ]
                        
                        options = [definition] + wrong_options
                        random.shuffle(options)
                        
                        return {
                            "question": question_text,
                            "options": options,
                            "correct_answer": definition,
                            "explanation": f"As stated in the text: '{sentence}'",
                            "type": "definition"
                        }
        
        return None
    
    def _generate_relationship_question(self, sentence: str, doc) -> Optional[Dict]:
        """Generate questions about relationships between concepts"""
        sent_doc = nlp(sentence)
        entities = [ent for ent in sent_doc.ents]
        
        if len(entities) >= 2:
            entity1, entity2 = entities[0], entities[1]
            
            # Find the relationship
            between_text = sentence[sentence.find(entity1.text) + len(entity1.text):sentence.find(entity2.text)].strip()
            
            question_text = f"What is the relationship between {entity1.text} and {entity2.text}?"
            
            correct_answer = f"{entity1.text} {between_text} {entity2.text}"
            
            wrong_options = [
                f"{entity1.text} has no relation to {entity2.text}",
                f"{entity2.text} controls {entity1.text}",
                f"{entity1.text} and {entity2.text} are the same thing"
            ]
            
            options = [correct_answer] + wrong_options
            random.shuffle(options)
            
            return {
                "question": question_text,
                "options": options,
                "correct_answer": correct_answer,
                "explanation": f"Based on the text: '{sentence}'",
                "type": "relationship"
            }
        
        return None
    
    def _generate_true_false_question(self, sentence: str, doc) -> Optional[Dict]:
        """Generate true/false questions"""
        # Sometimes make it true, sometimes false
        make_false = random.choice([True, False])
        
        if make_false:
            # Modify the sentence to make it false
            sent_doc = nlp(sentence)
            if sent_doc.ents:
                entity = random.choice([ent for ent in sent_doc.ents])
                # Replace with a different entity
                false_sentence = sentence.replace(entity.text, f"[INCORRECT_{entity.label_}]")
                question_text = f"True or False: {false_sentence}"
                correct_answer = "False"
                explanation = f"This is false. The correct statement is: '{sentence}'"
            else:
                # Negate the sentence
                false_sentence = "It is not true that " + sentence.lower()
                question_text = f"True or False: {false_sentence}"
                correct_answer = "False"
                explanation = f"This is false. The correct statement is: '{sentence}'"
        else:
            question_text = f"True or False: {sentence}"
            correct_answer = "True"
            explanation = f"This is true as stated in the text."
        
        return {
            "question": question_text,
            "options": ["True", "False"],
            "correct_answer": correct_answer,
            "explanation": explanation,
            "type": "true_false"
        }
    
    def _generate_question_from_chunk(self, chunk: str) -> Optional[Dict]:
        """Generate a question from a text chunk"""
        sentences = chunk.split('. ')
        if not sentences:
            return None
        
        sentence = random.choice([s for s in sentences if len(s) > 20])
        doc = nlp(chunk)
        
        # Try different question types
        question_funcs = [
            self._generate_factual_question,
            self._generate_true_false_question
        ]
        
        for func in question_funcs:
            try:
                question = func(sentence, doc)
                if question:
                    return question
            except:
                continue
        
        return None
    
    def _generate_wrong_options(self, correct: str, label: str, doc) -> List[str]:
        """Generate plausible wrong options"""
        wrong_options = []
        
        # Find entities of the same type
        similar_entities = [ent.text for ent in doc.ents if ent.label_ == label and ent.text != correct]
        wrong_options.extend(similar_entities)
        
        # Add some generic wrong options based on type
        if label == "PERSON":
            wrong_options.extend(["John Smith", "Jane Doe", "Dr. Johnson"])
        elif label == "ORG":
            wrong_options.extend(["Generic Corp", "Example Inc", "Test Organization"])
        elif label == "DATE":
            wrong_options.extend(["Last year", "Next month", "A decade ago"])
        elif label in ["LOC", "GPE"]:
            wrong_options.extend(["New York", "London", "Tokyo"])
        else:
            wrong_options.extend(["Something else", "Another option", "Different answer"])
        
        return list(set(wrong_options))[:5]
    
    def _validate_question(self, question: Dict, text: str) -> bool:
        """Validate that the question and answer are actually from the PDF"""
        # Check if the correct answer appears in the text
        return question["correct_answer"].lower() in text.lower()

# API Models
class QuizSession(BaseModel):
    session_id: str
    pdf_name: str
    theme_data: Dict
    questions: List[Dict]

class QuizSubmission(BaseModel):
    session_id: str
    user_name: str
    answers: Dict[int, str]

class QuizResult(BaseModel):
    score: int
    total_questions: int
    percentage: float
    detailed_results: List[Dict]
    user_name: str

# API Endpoints
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and generate quiz"""
    try:
        # Read file content
        content = await file.read()
        
        # Generate session ID
        session_id = hashlib.md5(content + str(datetime.now()).encode()).hexdigest()
        
        # Process PDF
        processor = PDFProcessor()
        text = processor.extract_text_from_pdf(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Chunk and index text
        chunks = processor.chunk_text(text)
        processor.create_embeddings(chunks)
        
        # Detect theme
        theme_detector = ThemeDetector()
        theme_data = theme_detector.get_theme_data(text, file.filename)
        
        # Generate questions
        quiz_gen = QuizGenerator(processor)
        questions = quiz_gen.generate_questions(text, num_questions=10)
        
        # Store session data
        conn = sqlite3.connect('quiz_results.db')
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO quiz_sessions (session_id, pdf_name, pdf_hash, upload_time, theme_data, quiz_data)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, file.filename, hashlib.md5(content).hexdigest(), 
              datetime.now(), json.dumps(theme_data), json.dumps(questions)))
        conn.commit()
        conn.close()
        
        return QuizSession(
            session_id=session_id,
            pdf_name=file.filename,
            theme_data=theme_data,
            questions=questions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get existing quiz session"""
    conn = sqlite3.connect('quiz_results.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT pdf_name, theme_data, quiz_data FROM quiz_sessions WHERE session_id = ?
    """, (session_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return QuizSession(
        session_id=session_id,
        pdf_name=result[0],
        theme_data=json.loads(result[1]),
        questions=json.loads(result[2])
    )

@app.post("/submit_quiz")
async def submit_quiz(submission: QuizSubmission):
    """Submit quiz answers and calculate score"""
    # Get quiz data
    conn = sqlite3.connect('quiz_results.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT quiz_data FROM quiz_sessions WHERE session_id = ?
    """, (submission.session_id,))
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        raise HTTPException(status_code=404, detail="Session not found")
    
    questions = json.loads(result[0])
    
    # Calculate score
    score = 0
    detailed_results = []
    
    for i, question in enumerate(questions):
        user_answer = submission.answers.get(str(i), None)
        is_correct = user_answer == question["correct_answer"]
        if is_correct:
            score += 1
        
        detailed_results.append({
            "question_index": i,
            "question": question["question"],
            "user_answer": user_answer,
            "correct_answer": question["correct_answer"],
            "is_correct": is_correct,
            "explanation": question["explanation"]
        })
    
    # Store results
    cursor.execute("""
        INSERT INTO quiz_results (session_id, user_name, score, total_questions, completion_time, detailed_results)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (submission.session_id, submission.user_name, score, len(questions),
          datetime.now(), json.dumps(detailed_results)))
    conn.commit()
    conn.close()
    
    return QuizResult(
        score=score,
        total_questions=len(questions),
        percentage=round((score / len(questions)) * 100, 1),
        detailed_results=detailed_results,
        user_name=submission.user_name
    )

@app.get("/sessions")
async def list_sessions():
    """List all available quiz sessions"""
    conn = sqlite3.connect('quiz_results.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT session_id, pdf_name, upload_time FROM quiz_sessions ORDER BY upload_time DESC
    """)
    sessions = cursor.fetchall()
    conn.close()
    
    return [
        {
            "session_id": s[0],
            "pdf_name": s[1],
            "upload_time": s[2]
        }
        for s in sessions
    ]

@app.get("/leaderboard/{session_id}")
async def get_leaderboard(session_id: str):
    """Get leaderboard for a specific quiz session"""
    conn = sqlite3.connect('quiz_results.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT user_name, score, total_questions, completion_time 
        FROM quiz_results 
        WHERE session_id = ?
        ORDER BY score DESC, completion_time ASC
        LIMIT 10
    """, (session_id,))
    results = cursor.fetchall()
    conn.close()
    
    return [
        {
            "rank": i + 1,
            "user_name": r[0],
            "score": r[1],
            "total_questions": r[2],
            "percentage": round((r[1] / r[2]) * 100, 1),
            "completion_time": r[3]
        }
        for i, r in enumerate(results)
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
