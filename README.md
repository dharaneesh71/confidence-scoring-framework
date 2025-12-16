# Confidence Scoring Framework for AI Outputs in Decision Support

## Overview
This project implements a Confidence Scoring Framework designed to evaluate and quantify the reliability of AI-generated answers in decision-support systems. Instead of relying solely on a language model’s response, the framework validates answers against a curated ground-truth knowledge base and produces a numerical confidence score along with supporting citations.

The system was developed as part of a Boeing-sponsored academic project with a focus on trust, explainability, and modular system design.

---

## Key Features
- AI-generated answers using a language model
- Validation against trusted, curated ground-truth documents
- Numerical confidence score (0.0 – 1.0)
- Transparent citations to supporting evidence
- Modular and scalable architecture
- Admin interface for uploading ground-truth PDFs

---

## System Architecture
The framework follows a layered and service-oriented architecture:

- **Frontend (React.js)**
  - User interface for submitting questions
  - Displays AI response, confidence score, and citations
  - Admin portal for uploading ground-truth documents

- **Backend (FastAPI)**
  - Orchestrates the entire workflow
  - Exposes APIs for query submission and document upload
  - Coordinates communication between AI model, vector database, and scoring pipeline

- **AI Model**
  - Uses LLaMA 3.1 as the answer generator
  - Treated only as a response generator (no self-scoring)

- **Knowledge Base**
  - Vector database built using ChromaDB
  - Stores embeddings of ground-truth documents
  - Supports semantic retrieval of relevant evidence

- **Confidence Scoring Pipeline**
  - Evaluates alignment between AI output and retrieved evidence
  - Uses semantic similarity, completeness, and precision metrics
  - Outputs a final confidence score with explanation

---

## Workflow
1. User submits a question through the frontend
2. Backend sends the question to the AI model
3. AI model generates a response
4. Backend retrieves relevant evidence from the knowledge base
5. Confidence scoring pipeline evaluates the response
6. Final output is returned with:
   - AI-generated answer
   - Confidence score
   - Supporting citations

---

## Confidence Score Definition
The confidence score represents the degree to which an AI-generated answer is supported by verified information in the ground-truth dataset.

- Confidence is not the same as accuracy
- Answers that explicitly state uncertainty are not penalized
- Scores reflect evidence support and factual grounding

---

## Technologies Used
- **Frontend:** React.js
- **Backend:** FastAPI (Python)
- **AI Model:** LLaMA 3.1
- **Vector Database:** ChromaDB
- **Embeddings:** Sentence-Transformers (MiniLM)
- **NLP Processing:** NLTK
- **Machine Learning:** Scikit-learn

---

## Accessibility & Usability
- WCAG 2.1 compliant UI design
- Keyboard-only navigation support
- Screen-reader friendly labels
- Clear system feedback for loading and errors

---

## Known Limitations
- Confidence quality depends on the coverage of the ground-truth dataset
- Retrieval errors may affect scoring accuracy
- Current evaluation is limited to top-k retrieved passages

---

## Testing
Testing has been conducted across multiple levels to ensure correctness and reliability:

- **Backend Testing**
  - API endpoint validation for query submission and document upload
  - Error handling for invalid inputs and missing data

- **Retrieval & Scoring Validation**
  - Verification of semantic search results from the vector database
  - Validation of confidence scores for correct, partial, and unsupported answers

- **Frontend Testing**
  - Manual testing of UI states (empty, loading, results, admin upload)
  - Accessibility and usability checks

- **End-to-End Testing**
  - Full pipeline testing from document upload to scored AI response

---

## Team
- Nipun  
- Jaideep  
- Dharaneesh
