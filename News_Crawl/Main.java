import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.*;

import org.json.JSONArray;
import org.json.JSONObject;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

public class Main {
    public static void main(String[] args) {
        System.out.println("heloooooooooooooooo sisisisiisiisiisisisisisiiiii in main ");
        Timer timer = new Timer();
        timer.schedule(new FetchNewsTask(), 0, 12 * 60 * 1000); // كل 10 دقائق = 600,000 ms
    }
}

class FetchNewsTask extends TimerTask {
    @Override
    public void run() {
        System.out.println("heloooooooooooooooo sisisisiisiisiisisisisisiiiii in run ");

        try {
            // String apiKey = "7830291514b44f048a3a8668e7cdcb44";
            // String apiKey = "7e52a84f16ab45a79d52d44610f81c0d";
            // String apiKey = "9d375cd8b10441369c46b764ab1cacd4";
            // String apiKey = "e40eeb4e154e44649ce92b41774bf10d";
            String apiKey = "3b2bebc1173a4b01abff531ae354ea7a";

            List<String> sources = Arrays.asList("cnn", "al-jazeera-english", "bbc-news", "associated-press");
            List<News> newsList = new ArrayList<>();

            for (String source : sources) {
                System.out.println("heloooooooooooooooo sisisisiisiisiisisisisisiiiii in inside for loop ");
                String urlStr = "https://newsapi.org/v2/top-headlines?sources=" + source + "&language=en&apiKey="
                        + apiKey;

                try {
                    URL url = new URL(urlStr);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("GET");

                    BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                    StringBuilder response = new StringBuilder();
                    String inputLine;

                    while ((inputLine = in.readLine()) != null) {
                        response.append(inputLine);
                    }
                    in.close();

                    JSONObject jsonResponse = new JSONObject(response.toString());
                    JSONArray articles = jsonResponse.getJSONArray("articles");

                    System.out.println("🔹 " + source + ": " + articles.length() + " articles");

                    for (int i = 0; i < articles.length(); i++) {
                        JSONObject article = articles.getJSONObject(i);

                        String title = article.optString("title", "No Title");
                        String content = article.optString("content", "No Content");
                        String publishedAtStr = article.optString("publishedAt", "");
                        String urlToArticle = article.optString("url", "No URL");
                        String imageUrl = article.optString("urlToImage", "No Image");

                        String sourceName = "Unknown";
                        if (article.has("source") && !article.isNull("source")) {
                            JSONObject sourceObj = article.getJSONObject("source");
                            sourceName = sourceObj.optString("name", "Unknown");
                        }

                        Date publishDate = null;
                        try {
                            publishDate = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.ENGLISH)
                                    .parse(publishedAtStr);
                        } catch (Exception e1) {
                            try {
                                publishDate = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.ENGLISH)
                                        .parse(publishedAtStr);
                            } catch (Exception e2) {
                                System.out.println("⚠️ Date parsing failed for: " + publishedAtStr);
                            }
                        }

                        String fullArticle = content;
                        try {
                            Document doc = Jsoup.connect(urlToArticle).userAgent("Mozilla").get();

                            Elements paragraphs = doc.select("article p, div p");
                            if (paragraphs.size() < 3) {
                                paragraphs = doc.select("p");
                            }

                            System.out.println("📄 Paragraphs found: " + paragraphs.size() + " from " + urlToArticle);

                            StringBuilder fullArticleBuilder = new StringBuilder();
                            for (Element paragraph : paragraphs) {
                                fullArticleBuilder.append(paragraph.text()).append("\n\n");
                            }

                            if (fullArticleBuilder.length() > 0) {
                                fullArticle = fullArticleBuilder.toString().trim();
                            }

                        } catch (Exception e) {
                            System.out.println("⚠️ Failed to fetch full content from: " + urlToArticle);
                        }

                        News news = new News(title, fullArticle, urlToArticle, sourceName, publishDate, imageUrl);
                        newsList.add(news);
                    }

                } catch (Exception e) {
                    System.out.println("❌ Failed to fetch from source: " + source);
                    e.printStackTrace();
                }
            }

            // Save to CSV
            try (PrintWriter writer = new PrintWriter(new FileWriter("news_live.csv"))) {
                writer.println("Title,Content,URL,Source,PublishedDate,ImageURL");

                for (News news : newsList) {
                    String row = String.format("\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\"",
                            news.getTitle().replace("\"", "'"),
                            news.getArticle().replace("\"", "'").replace("\n", " ").replace("\r", " "),
                            news.getUrl(),
                            news.getSource().replace("\"", "'"),
                            news.getPublishDate(),
                            news.getImageUrl());

                    writer.println(row);
                }

                System.out.println("✅ tatattatataa News saved to news_live.csv at " + new Date());

            } catch (Exception e) {
                System.out.println("❌ Failed to write to CSV file");
                e.printStackTrace();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}