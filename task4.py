import math
import re
from collections import Counter

MOVIES = {
    "3 Idiots": ("comedy drama", "college friends engineering education pressure friendship inspiring"),
    "Dangal": ("sports drama biography", "wrestling father daughters training champion determination"),
    "Lagaan": ("sports drama period", "cricket village tax british rural underdog team"),
    "Dilwale Dulhania Le Jayenge": ("romance drama", "love family europe wedding tradition young couple"),
    "Zindagi Na Milegi Dobara": ("comedy drama adventure", "friends road trip spain bachelor adventure self discovery"),
    "Queen": ("comedy drama", "travel woman independence honeymoon solo confidence self discovery"),
    "Drishyam": ("thriller crime drama", "family police secret murder cover up suspense"),
    "Andhadhun": ("thriller dark comedy music", "pianist blind twist murder suspense"),
    "Tumbbad": ("horror fantasy", "folklore greed curse treasure dark atmosphere"),
    "Gully Boy": ("music drama", "rap street dreams mumbai underdog talent struggle"),
    "Chak De India": ("sports drama", "hockey women team coach underdog national pride"),
    "PK": ("comedy sci-fi satire", "alien religion questions society humour message"),
    "Kahaani": ("thriller mystery", "missing husband investigation twist kolkata suspense"),
    "Taare Zameen Par": ("drama family", "child dyslexia teacher school emotional inspiring education"),
    "Rang De Basanti": ("drama patriotic", "college friends youth social change freedom fighters"),
    "Super 30": ("drama biography", "teacher maths poor students education struggle inspiring"),
    "Barfi": ("romance comedy drama", "love silent heartwarming friendship darjeeling"),
    "Wake Up Sid": ("drama romance comedy", "young man city growing up friendship coming of age mumbai"),
    "Bajrangi Bhaijaan": ("drama adventure", "child border journey kindness friendship emotional"),
    "Swades": ("drama", "scientist village return electricity social change inspiring"),
}

RATINGS = {
    "Aarav": {"Dangal": 5, "Lagaan": 5, "Chak De India": 5, "3 Idiots": 4, "Super 30": 4, "Swades": 4,
              "Rang De Basanti": 4, "Taare Zameen Par": 4, "Drishyam": 2, "Tumbbad": 1, "Andhadhun": 2},
    "Diya":  {"Dilwale Dulhania Le Jayenge": 5, "Barfi": 5, "Queen": 4, "Wake Up Sid": 4,
              "Zindagi Na Milegi Dobara": 5, "Bajrangi Bhaijaan": 4, "3 Idiots": 4, "Tumbbad": 1,
              "Kahaani": 2, "Dangal": 3},
    "Kabir": {"Drishyam": 5, "Andhadhun": 5, "Kahaani": 5, "Tumbbad": 4, "PK": 3, "Gully Boy": 3,
              "Dangal": 2, "Barfi": 2, "Dilwale Dulhania Le Jayenge": 2, "3 Idiots": 3},
    "Meera": {"3 Idiots": 5, "Taare Zameen Par": 5, "Super 30": 5, "Swades": 4, "Rang De Basanti": 4,
              "Bajrangi Bhaijaan": 4, "Dangal": 4, "Andhadhun": 2, "Tumbbad": 1, "Queen": 3},
    "Rohan": {"Gully Boy": 5, "Zindagi Na Milegi Dobara": 4, "Wake Up Sid": 5, "Queen": 4, "3 Idiots": 4,
              "Dangal": 3, "Kahaani": 2, "Lagaan": 3, "Barfi": 4, "PK": 4},
    "Sneha": {"Taare Zameen Par": 5, "Bajrangi Bhaijaan": 5, "Barfi": 4, "Dilwale Dulhania Le Jayenge": 4,
              "Drishyam": 3, "Swades": 4, "Super 30": 4, "Queen": 3, "Tumbbad": 2, "Chak De India": 3},
    "Vikram": {"Lagaan": 5, "Rang De Basanti": 5, "Swades": 5, "Chak De India": 4, "PK": 4, "Queen": 2,
               "Dangal": 5, "Super 30": 4, "Gully Boy": 3, "Barfi": 2},
    "Isha":  {"PK": 5, "3 Idiots": 5, "Andhadhun": 4, "Kahaani": 4, "Gully Boy": 3, "Lagaan": 3,
              "Drishyam": 4, "Tumbbad": 3, "Queen": 4, "Wake Up Sid": 3},
}

HYBRID_WEIGHT = 0.6   

def tokens(text):
    return re.findall(r"[a-z]+", text.lower())


def build_tfidf():
    docs = {t: tokens(g + " " + d) for t, (g, d) in MOVIES.items()}
    n = len(docs)
    df = Counter(w for words in docs.values() for w in set(words))
    idf = {w: math.log(n / c) + 1 for w, c in df.items()}
    vectors = {}
    for title, words in docs.items():
        tf = Counter(words)
        vectors[title] = {w: (c / len(words)) * idf[w] for w, c in tf.items()}
    return vectors


VECTORS = build_tfidf()


def cosine(a, b):
    dot = sum(a[w] * b.get(w, 0) for w in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def user_profile(ratings):
    profile = {}
    liked = {m: r - 3 for m, r in ratings.items() if r > 3}
    if not liked and ratings:                       # nothing above 3: use their best-rated movie
        best = max(ratings, key=ratings.get)
        liked = {best: 1}
    for movie, weight in liked.items():
        for w, val in VECTORS[movie].items():
            profile[w] = profile.get(w, 0) + weight * val
    return profile


def content_scores(ratings):
    profile = user_profile(ratings)
    cos = {m: cosine(profile, v) for m, v in VECTORS.items()}
    lo, hi = min(cos.values()), max(cos.values())
    span = (hi - lo) or 1
    avg = mean(ratings)
    return {m: max(1, min(5, avg + 3 * ((c - lo) / span - 0.5))) for m, c in cos.items()}


def content_score(ratings, movie):
    return content_scores(ratings)[movie]

def mean(ratings):
    return sum(ratings.values()) / len(ratings)


def similarity(a, b):
    common = set(a) & set(b)
    if len(common) < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    num = sum((a[m] - ma) * (b[m] - mb) for m in common)
    da = math.sqrt(sum((a[m] - ma) ** 2 for m in common))
    db = math.sqrt(sum((b[m] - mb) ** 2 for m in common))
    return num / (da * db) if da and db else 0.0


def cf_predict(ratings, movie, others):
    num = den = 0.0
    neighbours = 0
    for other in others.values():
        if movie in other:
            s = similarity(ratings, other)
            if s > 0:
                num += s * (other[movie] - mean(other))
                den += s
                neighbours += 1
    if den == 0:
        return None, 0
    return max(1, min(5, mean(ratings) + num / den)), neighbours

def hybrid_predict(ratings, movie, others):
    ct = content_score(ratings, movie)
    cf, n = cf_predict(ratings, movie, others)
    if cf is None:
        return ct, "content-based only"
    return HYBRID_WEIGHT * cf + (1 - HYBRID_WEIGHT) * ct, f"hybrid ({n} similar user(s))"


def explain(ratings, movie):
    liked = [m for m, r in ratings.items() if r >= 4]
    if not liked:
        return ""
    closest = max(liked, key=lambda m: cosine(VECTORS[movie], VECTORS[m]))
    return f"because you liked {closest}"


def recommend(ratings, others, top_n=5):
    results = []
    for movie in MOVIES:
        if movie in ratings:
            continue
        score, method = hybrid_predict(ratings, movie, others)
        results.append((movie, score, method, explain(ratings, movie)))
    return sorted(results, key=lambda x: x[1], reverse=True)[:top_n]


def similar_movies(title, top_n=4):
    scores = [(m, cosine(VECTORS[title], VECTORS[m])) for m in MOVIES if m != title]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


def evaluate():
    errors = {"content": [], "collaborative": [], "hybrid": []}
    total = 0
    for user, ratings in RATINGS.items():
        if len(ratings) < 3:
            continue
        others = {u: r for u, r in RATINGS.items() if u != user}
        for movie, true in ratings.items():
            rest = {m: r for m, r in ratings.items() if m != movie}
            total += 1
            errors["content"].append(abs(content_score(rest, movie) - true))
            cf, _ = cf_predict(rest, movie, others)
            if cf is not None:
                errors["collaborative"].append(abs(cf - true))
            errors["hybrid"].append(abs(hybrid_predict(rest, movie, others)[0] - true))
    print(f"Leave-one-out test on {total} ratings (lower MAE = better):")
    for name, errs in errors.items():
        if errs:
            print(f"  {name:14s} MAE {sum(errs) / len(errs):.2f}   (predicted {len(errs)}/{total})")


def print_recs(ratings, others):
    for movie, score, method, why in recommend(ratings, others):
        print(f"  {movie:30s} predicted {score:.1f}/5  [{method}] {why}")


def rate_new_user():
    titles = list(MOVIES)
    for i, t in enumerate(titles, 1):
        print(f"  {i:2d}. {t}")
    print("Enter 'number rating' (e.g. '3 5'). Rate at least 3 movies. Blank line to finish.")
    ratings = {}
    while True:
        line = input("rate> ").strip()
        if not line:
            if len(ratings) >= 3:
                return ratings
            print("Please rate at least 3 movies.")
            continue
        m = re.fullmatch(r"(\d+)\s+([1-5])", line)
        if m and 1 <= int(m.group(1)) <= len(titles):
            ratings[titles[int(m.group(1)) - 1]] = int(m.group(2))
        else:
            print("Format: movie-number rating(1-5)")


def main():
    print("Hybrid Bollywood Movie Recommender")
    while True:
        print("\n1) Recommendations for an existing user   2) I'm a new user (rate movies)")
        print("3) Movies similar to a title              4) Evaluate the methods     5) Quit")
        choice = input("> ").strip()
        if choice == "1":
            name = input(f"User ({', '.join(RATINGS)}): ").strip().capitalize()
            if name not in RATINGS:
                print("Unknown user.")
                continue
            others = {u: r for u, r in RATINGS.items() if u != name}
            print(f"{name} has rated: {RATINGS[name]}")
            print("Top picks:")
            print_recs(RATINGS[name], others)
        elif choice == "2":
            ratings = rate_new_user()
            print("Top picks for you:")
            print_recs(ratings, RATINGS)
        elif choice == "3":
            title = input("Movie title: ").strip().lower()
            match = next((m for m in MOVIES if m.lower() == title), None)
            if not match:
                print("Title not found.")
                continue
            for m, s in similar_movies(match):
                print(f"  {m:30s} similarity {s:.2f}")
        elif choice == "4":
            evaluate()
        elif choice == "5":
            break


if __name__ == "__main__":
    main()