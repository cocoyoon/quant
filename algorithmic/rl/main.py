from environment import Environment
from agent import Agent
from pandas_datareader import data


def main():

    df = data.DataReader("AAPL", "yahoo", "2020-01-01", "2021-01-01")
    env = Environment(chart_data=df)
    agent = Agent(environment=env)
    env.observe()

    print(agent.validate_action(action=0))


if __name__ == "__main__":

    main()
