import pandas as pd
import requests
import bs4
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import plotly
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"


nltk.download("stopwords")
plotly.offline.init_notebook_mode(connected=True)

# function http_client starts new session and setup the requests headers
def http_client():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent" : ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/106.0.0.0 Safari/537.36"),
            "Accept-Language" : "en-US, en:q=0.5"
        }
    )

    return session

# function make_request sends requests to amazon servers and returns response which can be converted to html
def make_request(client,product_id: str,page_number):
    baseurl = f"https://amazon.com/product-reviews/{product_id}/ref=cm_cr_getr_d_paging_btm_next_{page_number}?pageNumber={page_number}"
    try:
        response = client.get(baseurl)
    except requests.RequestException:
        raise Exception(f"HTTP Error")
        return
    if(response.status_code == 200):
        return response

# function scrape_reviews scrapes data from reviews page(username,date,review,rating)
def scrape_reviews(soup):
    data_list = [] # we'll store reviews in this list
    all_reviews = soup.find_all("div", class_="a-section review aok-relative") # locating all reviews

    for review in all_reviews:     # looping through all reviews and scraping data
        try:
            profile_name = review.find("span", class_="a-profile-name").text
        except:
            profile_name = ""
        try:
            title = review.find("a", {"data-hook": "review-title"}).findChild().text
        except:
            title = ""
        try:
            date = review.find("span", {"data-hook": "review-date"}).text
        except:
            date = ""
        try:
            stars = review.find("span", class_="a-icon-alt").text
        except:
            stars = ""
        try:
            review_text = review.find("span", {"data-hook": "review-body"}).findChild().text
        except:
            review_text = ""
        data = {"profile_name": profile_name,
                "title": title,
                "date": date,
                "stars": stars,
                "review": review_text} # storing data as dictionary, so we can easily convert it to DataFrame later
        data_list.append(data)
        print(data)

    df = pd.DataFrame(data_list)

    return df

# function extract_data loops through all pages on product reviews and makes recursive calls to scrape_reviews function. it returns DataFrame with all product reviews
def extract_data(client,product_id):
    pagination = 1 # we'll use this variable as pagination number. initial page number is 1 and it is increased from 1 to N
    has_next_page = True
    reviews_dfs = []
    while(has_next_page):
        response = make_request(client, product_id=product_id,page_number=pagination) # calling make_function to get Response
        pagination += 1 # increase page number by 1, so we don't scrape same page
        soup = bs4.BeautifulSoup(response.text,"lxml")
        current_page_data = scrape_reviews(soup=soup)
        print(len(current_page_data))
        reviews_dfs.append(current_page_data)
        if(len(current_page_data) == 0): # if there's no more data, we can stop trying going to next page by setting up has_next_page to False
            has_next_page = False
    df = pd.concat(reviews_dfs) # converting Dictionaries from List to DataFrame
    df.insert(0, 'New_ID', range(0, 0 + len(df)))
    df.index = df["New_ID"]
    return df

# function transform_data takes reviews DataFrame as input and it makes it clean
def transform_data(df):
    # Extract Date of Review, Rating Number, and Location of Review
    for index, row in df.iterrows():
        df.loc[index,"rating"] = float(row["stars"].split(" ")[0])
        df.loc[index,"location"] = row["date"].split("Reviewed in")[1].split("on")[0].strip()
        df.loc[index,"date_of_review"] = row["date"].split("on")[1].strip()
    df["date_format"] = pd.to_datetime(df["date_of_review"])

    # Clean Review Text
    stop_words = stopwords.words("english") # pull all English stopwords from NLTK library
    df["clean_review"] = df["review"].apply(lambda x: x.replace("\n","")) # remove \new lines
    df["clean_review"] = df["clean_review"].str.replace(r'[^\w\s]+','') # remove punctuation
    df["clean_review"] = df["clean_review"].str.replace(r'http\S+', '', regex=True)  # remove http links
    df["clean_review"] = df["clean_review"].str.lower() # converting text to lower-case
    df["clean_review"] = df["clean_review"].str.strip() # remove whitespace
    df["review_no_stopwords"] = df["clean_review"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words])) # remove stopwords

    df.drop(columns=["stars","date"],inplace=True) # Drop not needed columns
    print("dataframe successfully cleaned")
    return df

# function sentiment_analysis takes dataframe as input. it calculates polarity score for each review, and it marks it as positive/negative/neutral
def sentiment_analysis(df):
    # calculating polarity
    pol = lambda x: TextBlob(str(x)).sentiment.polarity  # function to calculate polarity
    df["polarity"] = df["clean_review"].apply(pol)  # apply polarity function to df

    # labeling as positive/negative/neutral
    df["positive"] = df["polarity"].apply(lambda x: "positive" if float(x) >= 0.01 else "")
    df["negative"] = df["polarity"].apply(lambda x: "negative" if float(x) <= -0.01 else "")
    df["score"] = df["positive"] + df["negative"]
    df["score"] = df["score"].replace("", "neutral")

    # labeling with boolean values
    df["positive"] = df["score"].apply(lambda x: True if x == "positive" else False)
    df["negative"] = df["score"].apply(lambda y: True if y == "negative" else False)
    df["neutral"] = df["score"].apply(lambda z: True if z == "neutral" else False)
    return df

# function sentiment_overtime builds interactive line plot showing number of positive/negative/neutral reviews overtime
def sentiment_overtime(df):
    df["month"] = df["date_format"].dt.month
    df["year"] = df["date_format"].dt.year
    df["date"] = pd.to_datetime(df["year"].astype(str)+ "-"+ df["month"].astype(str))
    grouped = df.groupby(["date"])["positive","negative","neutral"].sum().reset_index()
    grouped["neg%"] = grouped["negative"] / (grouped["negative"] + grouped["positive"] + grouped["neutral"]) * 100
    grouped["pos%"] = grouped["positive"] / (grouped["positive"] + grouped["negative"] + grouped["neutral"]) * 100
    grouped["neu%"] = grouped["neutral"] / (grouped["positive"] + grouped["negative"] + grouped["neutral"]) * 100
    grouped["total_reviews"] = grouped["positive"] + grouped["negative"] + grouped["neutral"]

    fig = px.line(grouped, x="date", y=grouped.columns[1:4], title='Product Review Sentiments Overtime')
    fig.show()

    return grouped


# calling functions
client = http_client()
product_id = "B00UCBJIG4"
result = extract_data(client=client,product_id=product_id)
clean_data = transform_data(result)
sentiments = sentiment_analysis(clean_data)
sentiments_overtime = sentiment_overtime(sentiments)
