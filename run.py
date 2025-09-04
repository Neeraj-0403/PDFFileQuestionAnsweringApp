import os 
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI 
from langchain.agents import initialize_agent, Tool
from datetime import datetime
import pytz
import yfinance as yf
  
load_dotenv()

OPENAI_MODEL="gpt-3.5-turbo"
OPENAI_API_KEY =os.environ.get("OPENAI_API_KEY")

def get_india_time(query: str) -> str:
    tz = pytz.timezone("Asia/Kolkata")
    india_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    return f"The current time in India is {india_time}"

def get_stock_price(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d")
    if data.empty:
        return f"Could not fetch data for {ticker}"
    latest_price = data["Close"].iloc[-1]
    return f"The current stock price of {ticker.upper()} is ${latest_price:.2f}"

def main():
    llm= ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name=OPENAI_MODEL)
    tool=[
        Tool(
            name="World Time Checker",
            func=get_india_time,
            description="Get the current time in major world cities"
        ),
         Tool(
            name="Stock Price Checker",
            func=get_stock_price,
            description="Get the current stock price for a given ticker symbol (like AAPL, TSLA, TCS.NS)"
        )
    ]

    # Create an agent with the tool
    agent = initialize_agent(tool, llm, agent="zero-shot-react-description", verbose=True)

    query = input("Ask your question: ")
    result = agent.run(query)
    print(result)

if __name__=="__main__":
    main()