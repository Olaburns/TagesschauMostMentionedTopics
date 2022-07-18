import requests
import datetime
from bertopic import BERTopic
import matplotlib.pyplot as plt
from matplotlib import rcParams

api_url = "https://www.tagesschau.de/api2/"

def get_dates(previous_days_count):
    dates = []
    format_data = "%y%m%d"
    for i in range(previous_days_count):
        date_unformatted = datetime.datetime.today() - datetime.timedelta(days=i)
        date = date_unformatted.strftime(format_data)
        dates.append(date)
    return dates

def get_titels(date):
    params = {"datumsangabe" : date}
    url = api_url + f"/newsfeed-101~_date-{date}.json"
    response = requests.get(url)
    news = response.json()["news"]
    titles = []
    for object in news:
        titles.append(object["title"])
    return titles
if __name__ == '__main__':
    previous_days_count = 25
    topic_count = 15

    dates = get_dates(previous_days_count)
    titles = []
    for date in dates:
        daily_titles = get_titels(date)
        titles.extend(daily_titles)

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(titles)

    df = topic_model.get_topic_info()

    #Drop first row with outliers
    df = df.drop(labels=0, axis=0)

    df.head(topic_count).plot.bar(x="Name", y="Count")
    fig = plt.gcf()
    fig.set_size_inches(10, 10, forward=True)
    plt.tight_layout()
    plt.show()


