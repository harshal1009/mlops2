text = """
MapReduce is simple. MapReduce is powerful.
"""

# --- Map step ---
mapped = []
for word in text.lower().replace('.', '').split():
    mapped.append((word, 1))

# --- Shuffle step ---
shuffled = {}
for word, count in mapped:
    if word not in shuffled:
        shuffled[word] = []
    shuffled[word].append(count)

# --- Reduce step ---
reduced = {}
for word, counts in shuffled.items():
    reduced[word] = sum(counts)

# --- Output ---
for word, count in reduced.items():
    print(f'{word}: {count}')
