import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import google.generativeai as genai

# Initialize Gemini client (replace 'your_api_key' with your actual API key)
genai.configure(api_key="your_api_key_here")

# Function to load CSV file
def load_csv():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

# Function to perform basic statistical analysis
def basic_statistics(df):
    # Select only numeric columns for correlation calculation
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    
    stats = {
        'mean': df[numeric_columns].mean().to_dict(),
        'median': df[numeric_columns].median().to_dict(),
        'mode': df[numeric_columns].mode().iloc[0].to_dict(),
        'std_dev': df[numeric_columns].std().to_dict(),
        'correlation': df[numeric_columns].corr().to_dict()
    }
    return stats

# Function to plot bar graph
def plot_bar(df, column, top_n=15):
    # Sort DataFrame by the specified column in descending order and select top N rows
    sorted_df = df.sort_values(by=column, ascending=False).head(top_n)
    
    # Check if 'Book Title' and 'Rating' columns exist in sorted_df
    if 'Book Title' not in sorted_df.columns or 'Rating' not in sorted_df.columns:
        st.write("Error: One or both of the columns 'Book Title' or 'Rating' not found in DataFrame.")
        return
    
    # Trim 'Book Title' to 15 characters
    sorted_df['Book Title'] = sorted_df['Book Title'].str.slice(0, 15)
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sorted_df['Book Title'], sorted_df['Rating'], color='blue')
    ax.set_xlabel('Book Title')
    ax.set_ylabel('Rating')
    ax.set_title(f'Top {top_n} Books by Rating')
    ax.set_xticklabels(sorted_df['Book Title'], rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Function to plot scatter graph
def plot_scatter(df, x_column, y_column):
    # Check if the specified columns exist in the DataFrame
    if x_column not in df.columns or y_column not in df.columns:
        st.write(f"Error: One or both of the columns '{x_column}' or '{y_column}' not found in DataFrame.")
        return
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[x_column], df[y_column], color='blue', alpha=0.5)
    ax.set_title(f'Scatter plot of {x_column} vs {y_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    plt.tight_layout()
    st.pyplot(fig)

# Function to plot line graph
def plot_line(df, x_column, y_column):
    # Check if the specified columns exist in the DataFrame
    if x_column not in df.columns or y_column not in df.columns:
        st.write(f"Error: One or both of the columns '{x_column}' or '{y_column}' not found in DataFrame.")
        return
    
    # Trim 'x_column' to 15 characters
    df[x_column] = df[x_column].astype(str).str.slice(0, 15)
    
    # Create the line plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df[x_column], df[y_column], marker='o', linestyle='-', color='blue', label=y_column)
    ax.set_title(f'Line plot of {y_column} over {x_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
# Function to generate response using Gemini API
def generate_llm_response(df, question):
    try:
        prompt = f"Dataset: {df.head(5).to_string()} \n\nQuestion: {question}\nAnswer:"
        response = genai.generate_text(prompt=prompt)
        
        # Check if response is valid and has 'text' attribute
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "No valid response found. Please check your query or try again later."
    
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit app
def main():
    st.title("TensorGo Assignment")
    
    # Load CSV
    df = load_csv()
    
    if df is not None:
        st.write("Dataframe Loaded Successfully:")
        st.write(df.head())

        # Display basic statistics
        st.subheader("Basic Statistics")
        stats = basic_statistics(df)
        st.write(stats)
        
        # Plot options
        st.subheader("Plots")
        
        # Bar Plot
        column_to_plot = st.selectbox('Select column for Bar Plot', df.columns)
        if st.button('Plot Bar Graph'):
            plot_bar(df, column_to_plot)
        
        # Scatter Plot
        x_column_scatter = st.selectbox('Select X column for Scatter Plot', df.columns)
        y_column_scatter = st.selectbox('Select Y column for Scatter Plot', df.columns)
        if st.button('Plot Scatter Graph'):
            plot_scatter(df, x_column_scatter, y_column_scatter)
        
        # Line Plot
        x_column_line = st.selectbox('Select X column for Line Plot', df.columns)
        y_column_line = st.selectbox('Select Y column for Line Plot', df.columns)
        if st.button('Plot Line Graph'):
            plot_line(df, x_column_line, y_column_line)
        
        # Generate LLM Response
        st.subheader("Ask a Question about the Data")
        question = st.text_input("Enter your question:")
        if st.button('Get Answer'):
            answer = generate_llm_response(df, question)
            st.write("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
