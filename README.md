# Advanced AI Text Detector

This project implements an "Advanced AI Text Detector" using Streamlit, designed to analyze text and determine the likelihood of it being AI-generated. It leverages various linguistic and machine learning models to assess different aspects of text, such as perplexity, burstiness, stylometry, Zipf's law distribution, and semantic drift.

## Features

-   **Text Analysis**: Input any text to get a comprehensive report on its AI likelihood.
-   **Multiple Metrics**: Analyzes text based on:
    -   **Perplexity**: Measures the predictability of the text.
    -   **Burstiness**: Examines the variation in sentence lengths.
    -   **Stylometry**: Assesses lexical diversity (Type-Token Ratio) and Part-of-Speech distribution.
    -   **Zipf's Law**: Compares word frequency distribution to the natural language pattern.
    -   **Semantic Drift**: Visualizes the semantic coherence and trajectory of sentences.
-   **Interactive Visualizations**: Provides Plotly charts for each metric to offer deeper insights.
-   **AI vs. Human Challenge**: A fun interactive section in the sidebar where users can guess which of two provided text samples is AI-generated and which is human-written.

## Setup and Installation

To get this project up and running, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Hw5_AdvancedTopic.git
    cd Hw5_AdvancedTopic
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    -   **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: Some dependencies like `torch` and `transformers` can be large and might take some time to install.*

## How to Run the Application

Once you have installed the dependencies, you can run the Streamlit application using the following command:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Project Structure

-   `app.py`: The main Streamlit application file, handling UI layout and orchestrating analysis.
-   `ui.py`: Contains functions for rendering UI elements, such as the sidebar and the AI vs. Human Challenge.
-   `analysis.py`: Implements the core text analysis algorithms (perplexity, burstiness, stylometry, etc.).
-   `plotting.py`: Contains functions for generating interactive plots using Plotly.
-   `requirements.txt`: Lists all Python dependencies required for the project.
-   `log.md`: (Optional) May contain development logs or notes.

Enjoy detecting AI-generated text!
