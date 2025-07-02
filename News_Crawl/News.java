import java.util.Date;

public class News {

    private String title;
    private String article;
    private Date publishDate;
    private String url;
    private String source;
    private String imageUrl;

    public News(String title, String article, String url, String source, Date publishDate, String imageUrl) {
        this.title = title;
        this.article = article;
        this.publishDate = publishDate;
        this.url = url;
        this.source = source;
        this.imageUrl = imageUrl;
    }

    public String getTitle() { return title; }
    public String getArticle() { return article; }
    public Date getPublishDate() { return publishDate; }
    public String getUrl() { return url; }
    public String getSource() { return source; }
    public String getImageUrl() { return imageUrl; }

    @Override
    public String toString() {
        return "Title: " + title + "\n"
                + "Content: " + article + "\n"
                + "URL: " + url + "\n"
                + "Source: " + source + "\n"
                + "Published Date: " + publishDate + "\n"
                + "Image: " + imageUrl + "\n"
                + "------------------------------";
    }
}