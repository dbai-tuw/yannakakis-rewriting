DROP TABLE IF EXISTS badges;
DROP TABLE IF EXISTS comments;
DROP TABLE IF EXISTS postHistory;
DROP TABLE IF EXISTS postLinks;
DROP TABLE IF EXISTS posts;
DROP TABLE IF EXISTS tags;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS votes;

CREATE TABLE badges (
    Id int NOT NULL,
    UserId int NOT NULL,
    Date date NOT NULL
);
CREATE TABLE comments (
    Id int NOT NULL,
    PostId int NOT NULL,
    Score int NOT NULL,
    CreationDate date NOT NULL,
    UserId int
);
CREATE TABLE postHistory (
    Id int NOT NULL,
    PostHistoryTypeId int NOT NULL,
    PostId int NOT NULL,
    CreationDate date NOT NULL,
    UserId int
);
CREATE TABLE postLinks (
    Id int NOT NULL,
    CreationDate date NOT NULL,
    PostId int NOT NULL,
    RelatedPostId int NOT NULL,
    LinkTypeId int NOT NULL
);
CREATE TABLE posts (
    Id int NOT NULL,
    PostTypeId int NOT NULL,
    CreationDate date NOT NULL,
    Score int NOT NULL,
    ViewCount int,
    OwnerUserId int,
    AnswerCount int,
    CommentCount int NOT NULL,
    FavoriteCount int,
    LastEditorUserId int
);
CREATE TABLE tags (
    Id int NOT NULL,
    Count int NOT NULL,
    ExcerptPostId int
);
CREATE TABLE users (
    Id int NOT NULL,
    Reputation int NOT NULL,
    CreationDate date NOT NULL,
    Views int NOT NULL,
    UpVotes int NOT NULL,
    DownVotes int NOT NULL
);
CREATE TABLE votes (
    Id int NOT NULL,
    PostId int NOT NULL,
    VoteTypeId int NOT NULL,
    CreationDate date NOT NULL,
    UserID int,
    BountyAmount int
);