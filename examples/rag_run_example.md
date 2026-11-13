# Example output: real RAG run

- Topic: Retrieval-Augmented Generation (RAG)
- Model: gemini-3.5-flash-lite
- Saved at: 2026-11-13T13:58:22+00:00
- Outcome: passed (6 of 6 checks passed)
- Attempts: 1; retries: 0

> This example was generated and evaluated against Gemini during a real local test. It documents that run only; future runs may receive different results.

## Lesson

## What you will learn

You will learn about a smart computer tool. 
It is called Retrieval-Augmented Generation, or RAG for short. 
It helps computer programs give better answers.

## The idea in simple words

Imagine you take a test. 
You are not allowed to use your memory alone. 
Instead, you can open a big book to find the correct facts. 
Then, you write down a smart answer using those facts. 

A computer program often works like this too. 
Sometimes, a computer program forgets new facts. 
It might even make up wrong answers. 
RAG fixes this problem. 
It lets the computer search a library of documents before it talks to you. 
This helps the computer give you true and helpful answers.

## How it works

RAG has two main steps. 
The first step is "retrieval," which means finding information. 
The second step is "generation," which means writing an answer.

First, you ask the computer a question. 
The computer does not just guess. 
It quickly searches through a private folder of text files or a database. 
A database is an organized collection of data saved on a computer. 
It finds the pages that match your question.

Second, the computer takes those pages and reads them. 
It uses that fresh text to write a clear answer just for you. 
It is like having a smart assistant who checks the rules first before speaking.

## Everyday examples

Here are two ways you might see RAG in real life.

**Example 1: A bank help chat**
You go to your bank's website. 
You type a question: "What are the rules for my credit card?" 
The bank's old computer program might guess the answer and get it wrong. 
With RAG, the computer first searches the newest bank rule book. 
It finds your exact credit card rules. 
Then, it writes a short and correct message to help you.

**Example 2: A doctor's study helper**
A doctor has a hard question about a rare illness. 
The doctor uses a special medical computer program. 
The program uses RAG. 
It searches thousands of new medical books and notes in seconds. 
It pulls out the newest facts about the illness. 
Then, it writes a summary for the doctor to read.

## Words to know

- **Database:** An organized collection of data or files saved on a computer.
- **Retrieve:** To find and bring back information from a storage place.
- **Generation:** The act of making or creating something, like writing an answer.

## Try this

Question: What is the first main step in RAG?

Answer: The first step is retrieval, which means finding the right information.

## Remember these points

- RAG helps computer programs give better and truer answers.
- It searches a database for facts before it writes a reply.
- It stops computers from guessing or making up wrong information.

## Actual structured evaluator results

```json
{
  "checks": {
    "accuracy_and_groundedness": {
      "result": "PASS",
      "reason": "The explanation of RAG is accurate, clear, and high-school appropriate without factual errors."
    },
    "beginner_friendly_language": {
      "result": "PASS",
      "reason": "Sentences are short, simple, and very easy to understand for a student with limited English."
    },
    "teaches_by_examples": {
      "result": "PASS",
      "reason": "It includes two concrete everyday examples (a bank help chat and a doctor's study helper) that show how RAG works."
    },
    "clear_language_no_unexplained_jargon": {
      "result": "PASS",
      "reason": "Technical terms like database, retrieval, and generation are clearly defined in plain language."
    },
    "coverage_of_key_points": {
      "result": "PASS",
      "reason": "It covers what RAG is, why it is needed, how it works, and provides examples and a summary."
    },
    "coherent_teaching_flow": {
      "result": "PASS",
      "reason": "The lesson follows a logical sequence from introduction, simple analogy, mechanism, examples, glossary, check, and summary."
    }
  },
  "failed_checks": [],
  "rejection_reasons": [],
  "overall": "PASS",
  "pass_count": 6,
  "overall_corrected": false
}
```

## Attempt log

- Attempt 1: PASS — 6 of 6 passed.
