import streamlit as st
import ollama
from pypdf import PdfReader
import json
import re
import time
import psutil
import platform


# ==================================================
# CONFIGURATION
# ==================================================

MODEL = "llama3.2:3b"


# ==================================================
# HELPERS
# ==================================================

def find_relevant_text(document, question, top_k=6):
    words = set(
        re.findall(r"\b\w+\b", question.lower())
    )

    chunks = [
        document[i:i + 2000]
        for i in range(0, len(document), 2000)
    ]

    if not chunks:
        return ""

    scored = []

    for chunk in chunks:
        chunk_words = set(
            re.findall(r"\b\w+\b", chunk.lower())
        )

        score = len(words & chunk_words)

        scored.append((score, chunk))

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = [
        chunk
        for score, chunk in scored[:top_k]
        if score > 0
    ]

    if selected:
        return "\n\n".join(selected)

    return document[:8000]


def extract_json(text):
    if not text:
        return None

    text = text.strip()

    # Remove markdown code fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Try complete response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        try:
            return json.loads(
                text[start:end + 1]
            )
        except json.JSONDecodeError:
            pass

    # Find JSON array
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end > start:
        try:
            return json.loads(
                text[start:end + 1]
            )
        except json.JSONDecodeError:
            pass

    return None


def clean_quiz(data, count):
    if isinstance(data, dict):
        questions = data.get(
            "questions",
            []
        )
    elif isinstance(data, list):
        questions = data
    else:
        return []

    if not isinstance(questions, list):
        return []

    cleaned = []

    for item in questions:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get(
                "question",
                ""
            )
        ).strip()

        options = item.get(
            "options",
            []
        )

        answer = item.get(
            "answer",
            0
        )

        if not question:
            continue

        if not isinstance(options, list):
            continue

        if len(options) < 4:
            continue

        options = [
            str(x).strip()
            for x in options[:4]
        ]

        if any(
            not x
            for x in options
        ):
            continue

        try:
            answer = int(answer)
        except (
            TypeError,
            ValueError
        ):
            answer = 0

        if answer < 0 or answer > 3:
            answer = 0

        cleaned.append({
            "question": question,
            "options": options,
            "answer": answer
        })

        if len(cleaned) >= count:
            break

    return cleaned


def generate_quiz(prompt, count):

    last_error = None

    for attempt in range(2):

        try:

            response = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json"
            )

            raw = (
                response
                .get("message", {})
                .get("content", "")
            )

            data = extract_json(raw)

            quiz = clean_quiz(
                data,
                count
            )

            if len(quiz) >= count:
                return quiz, None

            last_error = (
                "The model returned "
                "invalid or incomplete quiz data."
            )

        except Exception as exc:
            last_error = str(exc)

        prompt += """

IMPORTANT RETRY:

Return ONLY valid JSON.

Return exactly the requested
number of questions.

Every question must have
exactly four options.

Every answer must be
0, 1, 2, or 3.

Do not add explanations.
Do not add markdown.
"""

    return [], last_error


def analyze_study_performance(results):

    wrong_questions = [
        r for r in results
        if not r["is_correct"]
    ]

    correct_questions = [
        r for r in results
        if r["is_correct"]
    ]

    if not results:
        return None

    wrong_text = "\n\n".join(
        [
            f"""
Question:
{r["question"]}

Student answer:
{r["selected"]}

Correct answer:
{r["correct"]}
"""
            for r in wrong_questions
        ]
    )

    correct_text = "\n\n".join(
        [
            f"- {r['question']}"
            for r in correct_questions
        ]
    )

    if not wrong_questions:

        return {
            "summary": (
                "Excellent performance! "
                "No weak areas were detected."
            ),
            "strong_topics": [
                "All tested concepts"
            ],
            "weak_topics": [],
            "recommendation": (
                "Try a harder quiz or move "
                "to the next topic."
            )
        }

    prompt = f"""
You are an offline adaptive study coach.

Analyze the student's quiz performance.

Identify:
1. Strong concepts
2. Weak concepts
3. A short revision recommendation

Use ONLY the quiz questions and answers
provided below.

CORRECT QUESTIONS:
{correct_text}

WRONG QUESTIONS:
{wrong_text}

Return ONLY valid JSON in this format:

{{
    "summary": "short summary",
    "strong_topics": [
        "concept 1",
        "concept 2"
    ],
    "weak_topics": [
        "concept 1",
        "concept 2"
    ],
    "recommendation": "short study recommendation"
}}

Keep the answer concise.
"""

    try:

        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json"
        )

        raw = (
            response
            .get("message", {})
            .get("content", "")
        )

        data = extract_json(raw)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    return {
        "summary": (
            "You should revise the questions "
            "you answered incorrectly."
        ),
        "strong_topics": [
            "Concepts answered correctly"
        ],
        "weak_topics": [
            "Concepts from incorrect answers"
        ],
        "recommendation": (
            "Review the incorrect questions "
            "and attempt another quiz."
        )
    }


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Offline AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #4f46e5,
            #7c3aed
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
        opacity: 0.9;
    }

    .answer-box {
        padding: 25px;
        border-radius: 16px;
        background: #eef2ff;
        border-left: 5px solid #4f46e5;
        color: #111827;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>📚 Offline AI Study Assistant</h1>
        <p>
            Your personal AI tutor that works locally
            using Llama 3.2.<br>
            Ask questions, upload notes,
            test your knowledge,<br>
            get adaptive study guidance,
            and measure local AI performance.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# SESSION STATE
# ==================================================

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "coach_data" not in st.session_state:
    st.session_state.coach_data = None

if "coach_analysis" not in st.session_state:
    st.session_state.coach_analysis = None

if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("⚙️ Study Settings")

    st.write("### 📄 Study Material")

    uploaded_file = st.file_uploader(
        "Upload PDF (Optional)",
        type=["pdf"]
    )

    if uploaded_file is not None:

        try:

            reader = PdfReader(
                uploaded_file
            )

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += (
                        page_text + "\n"
                    )

            if text.strip():

                st.session_state.document_text = text

                st.success(
                    f"✅ PDF loaded\n\n"
                    f"{len(reader.pages)} pages"
                )

            else:

                st.session_state.document_text = ""

                st.warning(
                    "The PDF contains no "
                    "extractable text."
                )

        except Exception:

            st.session_state.document_text = ""

            st.error(
                "Could not read this PDF."
            )

    else:

        st.info(
            "No PDF uploaded.\n\n"
            "You can still ask normal "
            "study questions."
        )

    st.divider()

    st.write("### 🤖 AI Model")

    st.code(MODEL)

    st.caption(
        "Runs locally through Ollama."
    )


# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 AI Tutor",
    "📝 Quiz Mode",
    "🧠 Study Coach",
    "📊 Performance"
])


# ==================================================
# AI TUTOR
# ==================================================

with tab1:

    st.subheader(
        "💬 Ask Your AI Tutor"
    )

    if st.session_state.document_text:

        st.success(
            "📄 Your PDF is active. "
            "Questions can use information "
            "from your document."
        )

    else:

        st.info(
            "💡 You can ask questions normally, "
            "even without uploading a PDF."
        )

    question = st.text_area(
        "What do you want to learn?",
        placeholder=(
            "Example: Explain computer "
            "networks in simple language."
        ),
        height=120
    )

    col1, col2 = st.columns(2)

    with col1:

        ask_button = st.button(
            "🤖 Ask AI",
            use_container_width=True
        )

    with col2:

        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    if clear_button:

        st.rerun()

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            if st.session_state.document_text:

                relevant_text = find_relevant_text(
                    st.session_state.document_text,
                    question,
                    top_k=4
                )

                prompt = f"""
You are an offline AI study assistant.

Answer the student's question using the
relevant parts of the uploaded study material.

RELEVANT STUDY MATERIAL:

{relevant_text}

STUDENT QUESTION:

{question}

Instructions:

- Explain in simple language.
- Give an exam-friendly answer.
- Use headings and bullet points when useful.
- Prefer the provided study material.
- If the answer is not present in the provided
  material, clearly say so.
"""

            else:

                prompt = f"""
You are an offline AI study assistant.

STUDENT QUESTION:

{question}

Instructions:

- Explain in simple language.
- Give an exam-friendly answer.
- Use examples when useful.
- Break difficult concepts into simple steps.
"""

            with st.spinner(
                "🤖 AI is thinking..."
            ):

                try:

                    response = ollama.chat(
                        model=MODEL,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )

                    answer = (
                        response
                        .get("message", {})
                        .get("content", "")
                    )

                    if answer:

                        st.markdown(
                            '<div class="answer-box">',
                            unsafe_allow_html=True
                        )

                        st.subheader(
                            "🤖 AI Answer"
                        )

                        st.write(answer)

                        st.markdown(
                            "</div>",
                            unsafe_allow_html=True
                        )

                    else:

                        st.error(
                            "The AI returned "
                            "an empty answer."
                        )

                except Exception:

                    st.error(
                        "Could not connect to Ollama. "
                        "Make sure Ollama is running."
                    )


# ==================================================
# QUIZ MODE
# ==================================================

with tab2:

    st.subheader(
        "📝 Test Your Knowledge"
    )

    st.write(
        "Generate multiple-choice questions using AI."
    )

    if st.session_state.document_text:

        source_option = st.radio(
            "Quiz source",
            [
                "📄 My uploaded PDF",
                "🌐 General topic"
            ]
        )

    else:

        source_option = "🌐 General topic"

        st.info(
            "No PDF uploaded. "
            "The quiz will be generated "
            "from your topic."
        )

    topic = st.text_input(
        "📚 Enter topic",
        placeholder=(
            "Example: Computer Networks"
        )
    )

    number_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5
    )

    difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ]
    )

    generate_quiz_button = st.button(
        "🚀 Generate Quiz",
        use_container_width=True
    )

    if generate_quiz_button:

        if not topic.strip():

            st.warning(
                "Please enter a topic first."
            )

        else:

            if (
                source_option
                == "📄 My uploaded PDF"
            ):

                quiz_material = find_relevant_text(
                    st.session_state.document_text,
                    topic,
                    top_k=6
                )

                quiz_prompt = f"""
Create a multiple-choice quiz using ONLY
the study material below.

STUDY MATERIAL:

{quiz_material}

TOPIC:

{topic}

NUMBER OF QUESTIONS:

{number_questions}

DIFFICULTY:

{difficulty}

Rules:

- Create exactly {number_questions} questions.
- Every question must have exactly 4 options.
- Every answer must be an integer from 0 to 3.
- 0 = first option.
- 1 = second option.
- 2 = third option.
- 3 = fourth option.
- Do not use any other answer number.
- Do not include explanations.

Return exactly this JSON structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0
        }}
    ]
}}
"""

            else:

                quiz_prompt = f"""
Create a multiple-choice quiz about:

TOPIC:

{topic}

NUMBER OF QUESTIONS:

{number_questions}

DIFFICULTY:

{difficulty}

Rules:

- Create exactly {number_questions} questions.
- Every question must have exactly 4 options.
- Every answer must be an integer from 0 to 3.
- 0 = first option.
- 1 = second option.
- 2 = third option.
- 3 = fourth option.
- Do not use any other answer number.
- Do not include explanations.

Return exactly this JSON structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0
        }}
    ]
}}
"""

            with st.spinner(
                "🧠 Creating your quiz..."
            ):

                quiz_data, error_message = (
                    generate_quiz(
                        quiz_prompt,
                        number_questions
                    )
                )

            if quiz_data:

                st.session_state.quiz_data = quiz_data
                st.session_state.quiz_score = 0
                st.session_state.coach_data = None
                st.session_state.coach_analysis = None

                st.success(
                    f"🎉 Quiz generated! "
                    f"{len(quiz_data)} questions ready."
                )

            else:

                st.session_state.quiz_data = None

                st.error(
                    "The AI could not create "
                    "a valid quiz. Try 3 questions "
                    "on Easy difficulty."
                )

    # ==================================================
    # DISPLAY QUIZ
    # ==================================================

    if st.session_state.quiz_data:

        st.divider()

        st.subheader(
            "🎯 Your Quiz"
        )

        for i, q in enumerate(
            st.session_state.quiz_data
        ):

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### Question {i + 1}"
                )

                st.write(
                    q["question"]
                )

                st.radio(
                    "Choose your answer:",
                    q["options"],
                    key=f"question_{i}"
                )

        st.divider()

        submit_quiz = st.button(
            "🏆 Submit Quiz",
            use_container_width=True
        )

        if submit_quiz:

            score = 0
            results = []

            for i, q in enumerate(
                st.session_state.quiz_data
            ):

                selected = st.session_state.get(
                    f"question_{i}"
                )

                try:

                    correct_index = int(
                        q.get("answer", 0)
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    correct_index = 0

                # Safety check
                if (
                    correct_index < 0
                    or correct_index >= len(
                        q["options"]
                    )
                ):

                    correct_index = 0

                correct_answer = q["options"][
                    correct_index
                ]

                is_correct = (
                    selected == correct_answer
                )

                if is_correct:
                    score += 1

                results.append({
                    "question": q["question"],
                    "selected": (
                        selected
                        if selected
                        else "Not answered"
                    ),
                    "correct": correct_answer,
                    "is_correct": is_correct
                })

            total = len(
                st.session_state.quiz_data
            )

            percentage = (
                (score / total) * 100
                if total > 0
                else 0
            )

            # Save score
            st.session_state.quiz_score = score

            # Save data for Study Coach
            st.session_state.coach_data = {
                "score": score,
                "total": total,
                "percentage": percentage,
                "results": results
            }

            # Analyze performance
            with st.spinner(
                "🧠 Analyzing your performance..."
            ):

                st.session_state.coach_analysis = (
                    analyze_study_performance(
                        results
                    )
                )

            # Save history
            st.session_state.quiz_history.append({
                "score": score,
                "total": total,
                "percentage": percentage
            })

            st.success(
                f"🎉 Your Score: {score}/{total}"
            )

            if total > 0:

                if percentage >= 80:

                    st.balloons()

                    st.success(
                        "🔥 Excellent! "
                        "You are well prepared."
                    )

                elif percentage >= 50:

                    st.info(
                        "👍 Good job! "
                        "Revise the topics you missed."
                    )

                else:

                    st.warning(
                        "📖 Keep studying! "
                        "Try the Study Coach for revision."
                    )


# ==================================================
# ADAPTIVE STUDY COACH
# ==================================================

with tab3:

    st.subheader(
        "🧠 Adaptive Study Coach"
    )

    st.write(
        "Your offline AI coach analyzes your quiz "
        "performance and recommends what to study next."
    )

    if not st.session_state.coach_data:

        st.info(
            "📝 Complete a quiz first. "
            "Your Study Coach will appear here "
            "after you submit it."
        )

    else:

        coach_data = (
            st.session_state.coach_data
        )

        score = coach_data["score"]
        total = coach_data["total"]
        percentage = coach_data["percentage"]

        # Score section
        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🎯 Score",
                f"{score}/{total}"
            )

        with col2:

            st.metric(
                "📊 Accuracy",
                f"{percentage:.0f}%"
            )

        with col3:

            wrong_count = total - score

            st.metric(
                "❌ Need Revision",
                wrong_count
            )

        st.divider()

        analysis = (
            st.session_state.coach_analysis
        )

        if analysis:

            st.subheader(
                "📋 Coach Summary"
            )

            st.info(
                analysis.get(
                    "summary",
                    "Keep practicing."
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### 💪 Strong Areas"
                )

                strong_topics = analysis.get(
                    "strong_topics",
                    []
                )

                if strong_topics:

                    for topic_item in strong_topics:

                        st.success(
                            f"✓ {topic_item}"
                        )

                else:

                    st.write(
                        "Keep practicing to identify "
                        "your strongest concepts."
                    )

            with col2:

                st.markdown(
                    "### 📚 Weak Areas"
                )

                weak_topics = analysis.get(
                    "weak_topics",
                    []
                )

                if weak_topics:

                    for topic_item in weak_topics:

                        st.warning(
                            f"⚠️ {topic_item}"
                        )

                else:

                    st.success(
                        "No major weak areas detected."
                    )

            st.divider()

            st.subheader(
                "🎯 Recommended Next Step"
            )

            st.write(
                analysis.get(
                    "recommendation",
                    "Revise your incorrect answers."
                )
            )

        # Incorrect answers
        st.divider()

        st.subheader(
            "❌ Questions To Revise"
        )

        wrong_answers = [
            r
            for r in coach_data["results"]
            if not r["is_correct"]
        ]

        if wrong_answers:

            for index, item in enumerate(
                wrong_answers,
                start=1
            ):

                with st.expander(
                    f"Question {index}: "
                    f"{item['question']}"
                ):

                    st.write(
                        f"**Your answer:** "
                        f"{item['selected']}"
                    )

                    st.write(
                        f"**Correct answer:** "
                        f"{item['correct']}"
                    )

        else:

            st.success(
                "🎉 You got every question correct!"
            )

        # Targeted revision quiz
        st.divider()

        st.subheader(
            "🎯 Targeted Revision"
        )

        if wrong_answers:

            st.write(
                "Generate a new mini-quiz based "
                "only on the concepts you missed."
            )

            if st.button(
                "🚀 Generate Weak-Area Quiz",
                use_container_width=True
            ):

                wrong_text = "\n\n".join(
                    [
                        f"""
Question:
{item['question']}

Correct Answer:
{item['correct']}
"""
                        for item in wrong_answers
                    ]
                )

                revision_prompt = f"""
Create a 3-question multiple-choice
revision quiz.

Focus ONLY on the concepts represented
by these questions that the student
answered incorrectly:

{wrong_text}

Rules:

- Create exactly 3 questions.
- Every question must have exactly 4 options.
- Every answer must be an integer from 0 to 3.
- Questions should test the same concepts
  using different wording.
- Do not include explanations.

Return ONLY this JSON structure:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": 0
        }}
    ]
}}
"""

                with st.spinner(
                    "🧠 Creating your revision quiz..."
                ):

                    revision_quiz, error = (
                        generate_quiz(
                            revision_prompt,
                            3
                        )
                    )

                if revision_quiz:

                    st.session_state.quiz_data = (
                        revision_quiz
                    )

                    st.success(
                        "🎯 Revision quiz created! "
                        "Go to Quiz Mode to attempt it."
                    )

                else:

                    st.error(
                        "Could not create the "
                        "revision quiz."
                    )

        else:

            st.success(
                "🔥 You don't have any weak areas "
                "from this quiz. Try a harder quiz!"
            )


# ==================================================
# PERFORMANCE DASHBOARD
# ==================================================

with tab4:

    st.subheader(
        "📊 Performance Dashboard"
    )

    st.write(
        "Measure the resources used "
        "by your local AI model."
    )

    # MODEL INFORMATION

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    with col1:

        st.metric(
            "🤖 Model",
            "Llama 3.2 3B"
        )

    with col2:

        st.metric(
            "💾 Model Type",
            "Local"
        )

    with col3:

        st.metric(
            "🌐 Internet",
            "Not Required"
        )

    with col4:

        st.metric(
            "💻 AI Runtime",
            "Ollama"
        )

    st.divider()

    # HARDWARE INFORMATION

    st.subheader(
        "💻 Hardware Information"
    )

    hardware_col1, hardware_col2 = (
        st.columns(2)
    )

    with hardware_col1:

        st.metric(
            "🧠 Total RAM",
            f"{psutil.virtual_memory().total / (1024 ** 3):.1f} GB"
        )

        st.metric(
            "⚙️ CPU Cores",
            psutil.cpu_count(
                logical=True
            )
        )

    with hardware_col2:

        st.metric(
            "🖥️ Operating System",
            platform.system()
        )

        st.metric(
            "🏗️ Architecture",
            platform.machine()
        )

    st.divider()

    # PERFORMANCE TEST

    st.subheader(
        "⚡ Run Performance Test"
    )

    test_question = st.text_input(
        "Test Question",
        value=(
            "Explain what an operating "
            "system is."
        )
    )

    if st.button(
        "🚀 Run Performance Test",
        use_container_width=True
    ):

        memory_before = (
            psutil.virtual_memory().used
            / (1024 ** 3)
        )

        start_time = time.time()

        with st.spinner(
            "Testing local AI..."
        ):

            try:

                response = ollama.chat(
                    model=MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": test_question
                        }
                    ]
                )

                end_time = time.time()

                memory_after = (
                    psutil.virtual_memory().used
                    / (1024 ** 3)
                )

                response_time = (
                    end_time - start_time
                )

                memory_used = abs(
                    memory_after
                    - memory_before
                )

                st.success(
                    "✅ Performance test completed!"
                )

                result_col1, result_col2 = (
                    st.columns(2)
                )

                with result_col1:

                    st.metric(
                        "⚡ Response Time",
                        f"{response_time:.2f} seconds"
                    )

                with result_col2:

                    st.metric(
                        "🧠 RAM Change",
                        f"{memory_used:.2f} GB"
                    )

                st.subheader(
                    "🤖 Test Response"
                )

                st.write(
                    response
                    .get(
                        "message",
                        {}
                    )
                    .get(
                        "content",
                        "No response received."
                    )
                )

            except Exception:

                st.error(
                    "Could not run the performance "
                    "test. Make sure Ollama is running."
                )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "🔒 Offline AI Study Assistant • "
    "Powered by Python + Streamlit + "
    "Ollama + Llama 3.2"
)