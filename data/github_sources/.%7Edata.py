# # interests = [
# # (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
# # (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
# # (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
# # (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
# # (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
# # (3, "statistics"), (3, "regression"), (3, "probability"),
# # (4, "machine learning"), (4, "regression"), (4, "decision trees"),
# # (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
# # (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
# # (6, "probability"), (6, "mathematics"), (6, "theory"),
# # (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
# # (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
# # (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
# # (9, "Java"), (9, "MapReduce"), (9, "Big Data")
# # ]
# from collections import Counter
# from collections import defaultdict
# # # keys are interests, values are lists of user_ids with that interest
# # user_ids_by_interest = defaultdict(list)
# # for user_id, interest in interests:
# # 	user_ids_by_interest[interest].append(user_id)


# # interests_by_user_id = defaultdict(list)
# # for user_id, interest in interests:
# # 	interests_by_user_id[user_id].append(interest)



# # def most_common_interests_with(user):
# # 	return Counter(interested_user_id
# # 	for interest in interests_by_user_id[user["id"]]
# # 	for interested_user_id in user_ids_by_interest[interest]
# # 	if interested_user_id != user["id"])

# # print(most_common_interests_with({"id":0}))

# # salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
# # (48000, 0.7), (76000, 6),
# # (69000, 6.5), (76000, 7.5),
# # (60000, 2.5), (83000, 10),
# # (48000, 1.9), (63000, 4.2)]

# # def tenure_bucket(tenure):
# # 	if tenure < 2:
# # 		return "less than two"
# # 	elif tenure < 5:
# # 		return "between two and five"
# # 	else:
# # 		return "more than five"
# # salary_by_tenure = defaultdict(list)
# # for salary, tenure in salaries_and_tenures:
# # 	salary_by_tenure[tenure].append(salary)

# # salary_by_tenure_bucket = defaultdict(list)
# # for salary, tenure in salaries_and_tenures:
# # 	bucket = tenure_bucket(tenure)
# # 	salary_by_tenure_bucket[bucket].append(salary)

# # # keys are tenure buckets, values are average salary for that bucket
# # average_salary_by_bucket = {
# # tenure_bucket : sum(salaries) / len(salaries)
# # for tenure_bucket, salaries in salary_by_tenure_bucket.items()
# # }

# # print(average_salary_by_bucket)

# interests = [
# (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
# (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
# (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
# (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
# (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
# (3, "statistics"), (3, "regression"), (3, "probability"),
# (4, "machine learning"), (4, "regression"), (4, "decision trees"),
# (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
# (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
# (6, "probability"), (6, "mathematics"), (6, "theory"),
# (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
# (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
# (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
# (9, "Java"), (9, "MapReduce"), (9, "Big Data")
# ]

# words_and_counts = Counter(word
# for user, interest in interests
# for word in interest.lower().split())

# # print(words_and_counts)

# for word, count in words_and_counts.most_common():
# 	if count > 1:
# 		print( word, count)

# def lazy_range(n):
# 	i = 0
# 	print(i)
# 	while i< n:
# 		yield i
# 		print(i)
# 		i = i + 1

# for i in lazy_range(10):
# 	print("here")

def doubler(f):
	print(f)
	def g(x):sss
		print("hi")
		return 2*f(x)
	return g

def f1(x):
	return x+1

g = doubler(f1)
print(g(3))
from matplotlib import pyplot as plt
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]
# create a line chart, years on x-axis, gdp on y-axis
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
# add a title
plt.title("Nominal GDP")
# add a label to the y-axis
plt.ylabel("Billions of $")
plt.show() 