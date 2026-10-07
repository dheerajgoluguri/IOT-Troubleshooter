# 🔧 IoT Troubleshooter

> **AI-Powered Smart Device Troubleshooting Assistant**

IoT Troubleshooter is an AI-powered conversational assistant that helps users diagnose and resolve common Internet of Things (IoT) device problems.

The system understands a user's problem in natural language, retrieves relevant troubleshooting information from a knowledge base using **Retrieval-Augmented Generation (RAG)**, and provides clear, step-by-step guidance.

---

## 📌 Problem Statement

### Smart IoT Device Troubleshooting Chatbot

IoT devices are widely used in smart homes, industries, and connected environments. However, users often face problems such as:

- Wi-Fi connectivity failures
- Device configuration errors
- Pairing problems
- Firmware issues
- Device compatibility problems
- Devices going offline unexpectedly

Troubleshooting these problems often requires searching through manuals, forums, or customer-support resources.

### Our Solution

IoT Troubleshooter provides a conversational AI interface where users can simply describe their problem and receive:

- Problem identification
- Possible causes
- Relevant troubleshooting knowledge
- Step-by-step solutions
- Follow-up diagnostic questions

---

## 🎯 Key Features

### 💬 Natural-Language Troubleshooting
Users can describe their IoT problem in simple everyday language.

### 📚 RAG-Based Knowledge Retrieval
The system retrieves relevant troubleshooting information from an IoT knowledge base before generating the response.

### 🧠 AI-Powered Diagnosis
The AI analyzes the retrieved context and identifies possible causes of the reported problem.

### 🛠 Step-by-Step Solutions
Users receive structured troubleshooting instructions that are easy to follow.

### 🔄 Conversational Support
The assistant can ask follow-up questions and continue the conversation when additional information is required.

### 📡 Multiple IoT Devices
The system can help troubleshoot devices such as:

- Smart Bulbs
- Smart Plugs
- Smart Cameras
- Smart Speakers
- IoT Sensors

### ✨ Modern Web Interface
The application provides a clean and responsive chatbot interface with example problem prompts and interactive controls.

---

## 🏗️ System Architecture

```text
                    ┌──────────────┐
                    │     User     │
                    └──────┬───────┘
                           │
                     IoT Problem
                           │
                           ▼
                ┌────────────────────┐
                │   Flask Backend    │
                │      app.py        │
                └─────────┬──────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
      ┌────────────────┐      ┌─────────────────┐
      │ RAG Retrieval  │      │ Conversation    │
      │                │      │ History         │
      └───────┬────────┘      └────────┬────────┘
              │                        │
              ▼                        │
      ┌────────────────────┐           │
      │ IoT Knowledge Base │           │
      └─────────┬──────────┘           │
                │                      │
                └──────────┬───────────┘
                           ▼
                 ┌──────────────────┐
                 │   Groq API       │
                 │ GPT-OSS-120B     │
                 └────────┬─────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │ Troubleshooting     │
              │ Response            │
              └──────────┬──────────┘
                         │
                         ▼
                    Web Interface
