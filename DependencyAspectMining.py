import os, nltk, jieba, re, math
import urllib.parse, requests
import networkx as nx
from networkx.algorithms import isomorphism

def load(dianping="/home/jiakai/Data/Dianping/aspect/dianping.txt", max_valid=math.inf):
    with open(dianping, encoding="utf8") as reader:
        sentences, facts, lines = [], [], []
        counter, total = 0, 0
        
        for line in reader:
            line = line.strip()
            if line == "<DOC>" or line == "": continue

            # Cache
            if line != "</DOC>":
                lines.append(line)
                continue
            
            total += 1
            if len(lines) == 3:
                counter += 1
                sentences.append(lines[1])

                labels = []
                for pair in lines[2].split("\t"):
                    t = (pair[1:-1]).split(",")
                    l, r, p, o, n = t[0].strip(), t[1].strip(), int(t[2]), int(t[3]), t[4].strip() == "Y"

                    labels.append((l,r,p,o,n))
                
                facts.append(labels)
            lines = []
            if total == max_valid: break

        print("Data: %d reviews in total. %d(%.3f) reviews are useful, %d useless." % 
              (total, counter, counter/total, total-counter))
        return sentences, facts
    
    
def smaller(dianping="/home/jiakai/Data/Dianping/aspect/dianping.txt", 
            folder="/home/jiakai/Data/Dianping/aspect/", count=1000, splits=1):
    
    with open(dianping, encoding="utf8") as reader:
        lines, counter, split, writer = [], 0, 0, None
        
        for line in reader:
            line = line.strip()
            if line == "<DOC>" or line == "": continue

            # Cache
            if line != "</DOC>":
                lines.append(line)
                continue
                
            # Review without aspects
            if len(lines) != 3:
                lines = []
                continue
                
            if counter == 0:
                writer = open(folder + "dianping-1000-%d.txt" % (split + 1), "w")
            
            lines = [l+"\n" for l in lines]
            lines[0] = str(counter+1) + "\t" + lines[0]
            writer.writelines(["<DOC>\n"] + lines + ["</DOC>\n\n"])
            
            lines = []
            counter += 1
            if counter == count:
                writer.close()
                writer = None
                counter = 0
                split += 1
                
            if split == splits: break
        
        if writer: writer.close()
    
class Rule:
    def __init__(self, terms, pattern, extract):
        # First convert to a pattern graph
        self.pattern = nx.DiGraph()
        for i, term in enumerate(terms):
            self.pattern.add_node(i, pos=term[0], rel=term[1])
        for u,v in pattern:
            self.pattern.add_edge(u,v)
            
        # Extract
        self.extraction = extract
    
    def match(self, terms):
        # 
        g = nx.DiGraph()
        for t in terms: g.add_node(t[0], pos=t[3],  rel=t[5], val=t[1])
        for t in terms:
            if t[4]: g.add_edge(t[0], t[4])
        rg = g.reverse(copy=True)
        
        # Try match
        digm = isomorphism.DiGraphMatcher(g,self.pattern, node_match=Rule.is_match)
        
        pair = []
        for m in digm.subgraph_isomorphisms_iter():
            m = dict((v,k) for k, v in m.items())
            
            for e in self.extraction:
                li = [m[int(t)] for t in e[0]]
                ri = [m[int(t)] for t in e[1]]
                
                n = self.negation(g, rg, set(li+ri))
                n = sum(n.values())
                
                l = "".join(terms[t-1][1] for t in li)
                r = "".join(terms[t-1][1] for t in ri)
                
                pair.append((l,r,n))
        return pair
    
    nwords = re.compile("^[不没无非莫弗毋勿未否别無休]$")
    @classmethod
    def negation(cls, g, rg, ids):
        ns = dict(zip(ids, [0] * len(ids)))
        for i in ids:
            for _, v in g.edges_iter(i):
                if cls.nwords.search(g.node[v]["val"]):
                    ns[i] += 1
            for _, v in rg.edges_iter(i):
                if cls.nwords.search(rg.node[v]["val"]):
                    ns[i] += 1
        return ns
        
    @classmethod
    def is_match(cls, u, v):
        """
        Match the node
        """
        if not v["pos"].search(u["pos"]): return False
        if not v["rel"].search(u["rel"]): return False
        return True
    
    @classmethod
    def load_rules(cls, file):
        import json
        
        Terms = lambda terms: [(re.compile(t[0]), re.compile(t[1])) for t in terms]
        
        with open(file) as reader:
            rules = json.load(reader)                
            return [Rule(Terms(r["t"]), r["p"], r["e"]) for r in rules]
        
class Extractor:
    
    level_subsen = re.compile("[,?!.，？！．。 ]|\\n")
    level_sen = re.compile("[?!.？!．。]|\\n")
    
    patterns = [("a","n"),("n","a"),("v","a")]
    
    url = "http://localhost:8080/hanlp/"
    segment = url + "segment"
    dependency = url + "dependency"
    keywords = url + "keywords"
    phrase = url + "phrase"
    summary = url + "summary"
    
    @classmethod
    def get(cls, url, **params):
        try:
            #quoted = dict((k, urllib.parse.quote(v)) for k, v in params.items())
            r = requests.get(url, params=params)

            if r.status_code == 200:
                return r.json()

        except (ConnectionError, Timeout) as e:
            pass

        return None
    
    @classmethod
    def extract_terms(cls, sentences, level, method, **params):
        terms = []
        for sentence in sentences:
            senterms = []
            for sen in level.split(sentence):
                if not sen: continue
                t = cls.get(method, sentence=sen, **params)
                if not t: continue
                print(sen)
                print()
                print(t)
                print()
                print()
                senterms.append(t)
            terms.append(senterms)
        return terms

    @classmethod
    def extract_by_rules(cls, terms, rules):
        for rule in rules:
            pairs = rule.match(terms)
            if pairs: return pairs
        return []
    
    @classmethod
    def co_occurrence(cls, pairs):
        co = nx.Graph()
        for u, v, _ in pairs:
            if co.has_edge(u, v):
                co[u][v]["w"] += 1
            else:
                co.add_edge(u, v, w=1)
            co.node[u]["w"] = co.node[u].get("w", 0) + 1
            co.node[v]["w"] = co.node[v].get("w", 0) + 1
        return co
    
    @classmethod
    def co_by_rules(cls, terms, rules):
        pairs = []
        for senterms in terms:
            for term in senterms:
                pair = Extractor.extract_by_rules(term, rules)
                if pair: pairs.extend(pair)
        co = cls.co_occurrence(pairs)
        return co
    
    @classmethod
    def idf(cls, terms):
        df = {}
        for senterms in terms:
            for term in senterms:
                do = set()

                for t in term:
                    if t[1] in do: continue
                    do.add(t[1])

                    df[t[1]] = df.get(t[1], 0) + 1
        return dict((k, v/len(terms)) for k, v in df.items())

class AspectMining:
    # Basic PMI definition
    pmi_basic = lambda x, y, xy, n: math.log(n * xy / (x * y))
    # Expected value of PMI
    pmi_avg = lambda x, y, xy, n: (xy/n) * AspectMining.pmi_basic(x, y, xy, n)
    # KL distance, with x be an aspect, which is more reliable
    pmi_kl = lambda x, y, xy, n: (xy/x) * AspectMining.pmi_basic(x, y, xy, n)
    # basic pmi weighted with newly identified term frequency
    pmi_tf = lambda x, y, xy, n: (y/n) * AspectMining.pmi_basic(x, y, xy, n)
    
    @classmethod
    def pmi_all(cls, co, func, tco=0):
        """
        For a specified term, calculate the specified version of Point-wise Mutual Information.
        @param tco: Threshold for co-occurrence
        """
        n = co.number_of_nodes()
        return [(u, v, func(co.node[u]["w"], co.node[v]["w"], xy, n)) for u, v, xy in co.edges_iter(data="w") if xy > tco]
    
    @classmethod
    def pmi(cls, co, term, func, tco=0):
        """
        For a specified term, calculate the specified version of Point-wise Mutual Information.
        @param tco: Threshold for co-occurrence
        """
        if not co.has_node(term):
            return []
        
        n = co.number_of_nodes()
        x = co.node[term]["w"]
        
        return [(v, func(x, co.node[v]["w"], xy, n)) for u, v, xy in co.edges_iter(term, data="w") if xy > tco]
    
    @classmethod
    def lookinto(cls, co, term, func, n=math.inf, tco=0, tv=0):
        """
        Look into how the pmi value is calculated
        @param tco: Threshold for co-occurrence
        @param tv: Threshold for PMI value
        """
        pmis = cls.pmi(co, term, func, tco)
        pmis = sorted(pmis, key=lambda x: x[1], reverse=True)
        
        nodes = co.number_of_nodes()
        for i, pmi in enumerate(pmis):
            if i == n or pmi[1] <= tv:
                break
            X, Y = term, pmi[0]
            print("For <n=%d, X=\'%s\',Y=\'%s\'>: PMI=%.5f, x=%d, y=%d, xy=%d" % 
                  (nodes, X, Y, pmi[1], co.node[X]["w"], co.node[Y]["w"], co[X][Y]["w"]))
    
    @classmethod
    def ascore(cls, co):
        """
        Implement the a-score statistic in:
        Unsupervised domain-independent aspect detection for sentiment analysis of customer reviews
        
        Not good enough
        """
        scores = {}
        n = co.number_of_nodes()
        for u, x in co.nodes_iter(data="w"):
            x = x["w"]
            score = 0
            for _, v, xy in co.edges_iter(u, data="w"):
                y = co.node[v]["w"]
                score += math.log(1 + xy*n/(x*y))
            scores[u] = x * score / n
        return scores

    @classmethod
    def coidf(cls, co, idf):
        """
        Inspired by TF-IDF with TF replaced by the number of distinct co-occurrence term.
        
        Not good enough
        """
        coidf = [(u, len(co[u])/idf[u]) for u in co.nodes_iter()]
        return sorted(coidf, key=lambda x: x[1], reverse=True)
    
    @classmethod
    def tfidf(cls, co, idf):
        ti = [(u, co.node[u]["w"]) for u in co.nodes_iter()]
        return sorted(ti, key=lambda x: x[1], reverse=True)
    
    @classmethod
    def boosting(cls, co, seeds, pmifunc, n, tco=0, decay=1):
        """
        @param pmifunc: the pmi function
        @param n: select at most `n` words
        @param tco: Threshold for co-occurrence
        @param decay: The reliability of a term
        """
        import queue
        
        # A boosted term is less reliable if it is boosted from later iterations
        iterations = dict(zip(seeds,[0] * len(seeds)))
        
        def insertq(q, co, seed):
            pmis = cls.pmi(co, seed, pmifunc, tco)
            for v, w in pmis:
                # Decay
                q.put((-w * decay ** iterations[seed], seed, v))
        
        q = queue.PriorityQueue()
        for seed in seeds:
            insertq(q, co, seed)
        
        boost = []
        while len(boost) <= n and q.qsize() > 0:
            w, u, v = q.get()

            if v in iterations.keys():
                continue
            
            iterations[v] = iterations[u] + 1 
            
            boost.append((len(boost)+1, u, v, -w, decay, iterations[u]))

            insertq(q, co, v)
            
        return boost
    
    @classmethod
    def filters(cls, arts, pos, filters):
        terms = set()
        for art in arts: terms.add(art[1]), terms.add(art[2])
        
        return [t for t in terms if t in pos.keys() and filters.search(pos[t])]

class Polarity:
    """
    Load semantics including:
    1. Word polarity value (1 for positive and -1 for negative) based on Tsinghua word polarity.
    2. Synonym words based on HIT synonym words
    """
    def __init__(self, pfiles, nfiles, sfile, overrides):
        self.load_polarity(pfiles, nfiles)
        self.polarity.update(overrides)
        self.load_synonym(sfile)
        self.propagate_polarity()
        
    def load_polarity(self, pfiles, nfiles):
        """
        Load Tsinghua word polarity
        """
        
        def __load(files, polarity):
            ret = {}
            for file, encoding in files:
                with open(file, encoding=encoding) as reader:
                    ret.update((w.strip(), polarity) for w in reader if not w.startswith("#"))
            return ret
        
        self.polarity, n = __load(pfiles, 1), __load(nfiles, -1)
        self.polarity.update((k, -1) if k not in self.polarity.keys() else (k,0) for k in n.keys())
    
    @property
    def ambiguous(self):
        """
        Return words with ambiguous polarity
        """
        return set(k for k, v in self.polarity.items() if v ==0)
    
    @property
    def positive(self):
        return set(k for k, v in self.polarity.items() if v == 1)
    
    @property
    def negative(self):
        return set(k for k, v in self.polarity.items() if v == -1)
    
    def load_synonym(self, sfile):
        """
        Load HIT synonym words
        """
        self.synonym = nx.Graph()
        
        with open(sfile, encoding="gbk") as reader:
            for line in reader:
                words = line.split()
                if len(words) < 3: continue
                    
                for w in words[1:]: self.synonym.add_edge(w, words[0])        
    
    def slookup(self, w):
        """
        Lookup synonym for `w`
        """
        if not self.synonym.has_node(w):
            return set()
        
        synonym = set()
        for _, sid in self.synonym.edges_iter(w):
            for _, s in self.synonym.edges_iter(sid):
                if s == w:
                    continue
                synonym.add(s)
        return synonym
    
    def propagate_polarity(self):
        for k, v in self.polarity.items():
            if not self.synonym.has_node(k): continue
                
            for _, sid in self.synonym.edges_iter(k):
                if not self.synonym.node[sid].get("polarity"):
                    self.synonym.node[sid]["polarity"] = v
                else:
                    self.synonym.node[sid]["polarity"] += v
                    
        for u, data in self.synonym.nodes_iter(data="polarity"):
            if not data:
                continue
            data["polarity"] /= len(self.synonym[u])
    
    def splookup(self, w):
        """
        Lookup dominant polarity of `w`'s synonym
        """
        if not self.synonym.has_node(w):
            return None
        polarity, found, counter = 0, False, 1
        for _, sid in self.synonym.edges_iter(w):
            p = self.synonym.node[sid].get("polarity")
            if p: polarity, found = polarity + p, True
            
            counter += 1
        
        return polarity/counter if found else None
    
    def plookup(self, w):
        """
        Lookup word polarity
        """
        p = self.polarity.get(w)
        return p if p else self.splookup(w)
            
        
def evaluate(fact, pairs):
    print(fact)
    print(pairs)
    print()
    # Numbers:
    # 0: number of facts
    # 1: number of correctly identified left word
    # 2: number of correctly identitied left and right word
    # 3: number of correctly identified left word, right word and polarity
    # 4: number of identified pairs
    ret = [len(fact), 0, 0, 0, len(pairs)]
    if not pairs: return ret
    
    idx = {}
    for l, r, p, t, n in fact:
        if l not in idx.keys():
            idx[l] = {r: p}
        if r not in idx[l].keys():
            idx[l][r] = p
    
    for l, r, p, n in pairs:
        if l in idx.keys():
            ret[1] += 1
        else:
            continue
        
        if r in idx[l].keys():
            ret[2] += 1
        else:
            continue
        
        if p is None: continue
        p = -p if n % 2 else p
        p = 1 if p > 0 else -1
        if p == idx[l][r]:
            ret[3] += 1
    
    return ret

def validate(data, facts, aspects, polarity, rules):
    # words of interest, including synonym
    woi = set()
    for a in aspects:
        lsyn = polarity.slookup(a[1])
        rsyn = polarity.slookup(a[2])
        woi.update(lsyn)
        woi.update(rsyn)
    
    stats = []
    for sen, fact in zip(data, facts):
        subsens = Extractor.extract_terms([sen], Extractor.level_subsen, Extractor.dependency)
        pairs = []
        for subsen in subsens[0]:
            extracted = Extractor.extract_by_rules(subsen, rules)
            for l, r, neg in extracted:
                if l not in woi or r not in woi: continue
                    
                # Now calculate
                p = polarity.plookup(r)
                if not p: p = polarity.splookup(r)
                pair = (l,r,p,neg)
                pairs.append(pair)
        e = evaluate(fact, pairs)
        stats.append(e)
    return stats

pfiles = [("/home/jiakai/Data/SentimentAnalysisWordsLib/tsinghua.positive.gb.txt","gbk"),
          ("/home/jiakai/Data/SentimentAnalysisWordsLib/NTUSD_positive_simplified.txt","utf_16_le")]
nfiles = [("/home/jiakai/Data/SentimentAnalysisWordsLib/tsinghua.negative.gb.txt","gbk"),
          ("/home/jiakai/Data/SentimentAnalysisWordsLib/NTUSD_negative_simplified.txt","utf_16_le")]
sfile = "/home/jiakai/Data/SentimentAnalysisWordsLib/哈工大/同义词词林扩展版.txt"
overrides = [("高",-1),("贵",-1)]
#polarity = Polarity(pfiles, nfiles, sfile, overrides)   
#rules = Rule.load_rules("/home/jiakai/Codes/Aspect Mining/rules.json")
#rules = [rules[4]]

#data, facts = load(max_valid=200)
terms = Extractor.extract_terms(data, Extractor.level_sen, Extractor.dependency)
#co = Extractor.co_by_rules(terms, rules)
#tfidf = AspectMining.tfidf(co, Extractor.idf(terms))
#seeds = set([t[0] for t in tfidf[:10]])
#aspects = AspectMining.boosting(co, seeds, AspectMining.pmi_avg, 200, 3, 0.9)
