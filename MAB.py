import re, queue, math
import multiprocessing as mp
import threading
import jieba.posseg as psg
import logging, sys

logging.basicConfig(format='%(asctime)s:[Thread-%(thread)s]%(message)s',level=logging.DEBUG, stream=sys.stdout)


class Extractor:
    """
    Extract terms from sentences
    """
    
    # Nouns, verbs, adjectives and adverbs
    filters = re.compile("(^n)|(^v$)|(^ad?$)")
    puncs = re.compile("$[xw]")
    
    # commas
    commas = re.compile(" , ，、")
    
    # To segment sub-sentences
    periods = re.compile("[?!.？!．。]|\\n")
        
    ###############################################################################
    # Extract terms
    ###############################################################################
    @classmethod
    def innersentences(cls, sentence):
        return cls.commas.split(sentence)
    
    @classmethod
    def subsentences(cls, sentence):
        return cls.periods.split(sentence.strip())
    
    @classmethod
    def frequent(cls, terms, n, threshold=math.inf):
        words = []
        for sen, ts in terms:
            ws = [t for t in ts if not isinstance(t, int)]
            for w in ws:
                words.extend(w)
        
        from collections import Counter
        frqs = Counter(words)
        frqs = sorted(frqs.items(), key=lambda x: x[1], reverse=True)
        return [(w, f) for i, (w, f) in enumerate(frqs) if f < threshold and i < n]
    
    @classmethod
    def extract(cls, sentence):
        if not sentence:
            return []
        
        terms, concat, nonpuncs = [], [], 0
        for t, pos in psg.cut(sentence):
            if cls.filters.search(pos):
                if nonpuncs:
                    terms.append(nonpuncs)
                    nonpuncs = 0
                concat.append(t)
            else:
                if concat:
                    terms.append(concat)
                    concat = []
                if not cls.puncs.search(pos):
                    nonpuncs += 1
        if concat:
            terms.append(concat)
            
        return terms
    
    @classmethod
    def single(cls, sentence):
        """
        Extract terms of interest for all sub-sentences of a sentence
        """
        result = []
        
        # For each sub-sentence
        for sen in cls.subsentences(sentence):
            terms = cls.extract(sen)
                
            if terms:
                result.append((sen, terms))
                
        return result
    
    @classmethod
    def batch(cls, sentences, multithreads=True):
        """
        Extract terms for all sentences
        """
        rqueue = queue.Queue()
        
        def __extract(sentences):
            """
            Extract terms of interest for a list of sentences.
            """
            logging.info("Enter thread for extracting terms.")
            result = []
            for sentence in sentences:
                ret = cls.single(sentence)
                
                if ret:
                    result.extend(ret)
                    
            if result:
                rqueue.put(result)
                
            logging.info("Exit thread for extracting terms.")
    
        if not multithreads:
            __extract(sentences)
            
        else:
            from multiprocessing import cpu_count
            # First extract terms
            threads = []
            for split in cls.slices(len(sentences), cpu_count() * 2):
                thread = threading.Thread(target=__extract,args=(sentences[split],))
                thread.start()
                threads += [thread]
                
            for i, thread in enumerate(threads):
                thread.join()
        
        # Extracted
        combined = []
        while not rqueue.empty():
            item = rqueue.get()
            combined.extend(item)
            
        return combined
    
    @classmethod
    def slices(cls, total, splits):
        """
        Split `[0,total)` to `splits` slices
        """
        ret = []
        start, step = 0, (total + splits - 1) // splits
        while True:
            end = min(start + step, total)
            ret.append(slice(start, end))
            if end == total:
                break
            start = end
        return ret
    
    
class ART:
    """
    Aspect Related Term mining
    """
    def __init__(self, aspects, n, m, k):
        """
        @param aspects: A dictionary of aspect->seeds
        @param sentences: a list of text
        @param n: select n terms with maximum cvalue
        @param m: select m aspect related terms for each aspect
        @param k: the context to search cooccurrence
        """
        self.__aspects = aspects
        self.__n = n
        self.__m = m
        self.__k = k
    
    ###############################################################################
    # Calculate term frequency, term inclusion and term cooccurrence
    ###############################################################################
    def substrings(self, terms, size):
        step = 1
        # The term itself is not included
        while step < size:
            start, end = 0, step 
            while end <= size:
                yield ("".join(terms[start:end]), step)
                start, end = start + 1, end + 1
            step += 1
            
    def terms_info(self, terms_shorten):
        """
        Return 1) term->(size,frequency) 2) term->contained-by-terms
        """
        logging.info("Calculating term frequency and inclusion information.")

        def __frequency(frqs, term, size):
            """
            Increase frequency
            """
            # tinfo: term->(freq, size)
            if term not in frqs.keys():
                frqs[term] = (size, 1)
            else:
                frqs[term] = (size, frqs[term][1] + 1)

        tinfo, contains = {}, {}

        # Collect multi-terms size and frequency information
        for sen, shorten in terms_shorten:
            for terms in [t for t in shorten if not isinstance(t, int)]:
                term, size = "".join(terms), len(terms)

                __frequency(tinfo, term, size)

                # Classify terms by term number
                for s, ss in self.substrings(terms, size):
                    __frequency(tinfo, s, ss)

                    if s not in contains.keys():
                        contains[s] = set([term])
                    else:
                        contains[s].add(term)

        return tinfo, contains
    
    def terms_cooccurrence(self, aspect, seeds, terms, arts):
        """
        Search co-occurrence between targets and ART
        """
        def __search(co, tois, words, i, pos, k):
            """
            @param pos: position operator
            """
            counter, j = 0, 1
            while 0 <= pos(i, j) and pos(i, j) < len(words) and counter <= k:
                word = words[pos(i, j)]
                # An integer
                if isinstance(word, int):
                    counter += word
                else:
                    if word[0] in tois:
                        key, v = words[i][0], word[0]
                        # Add to co-occurrence
                        if key not in co.keys():
                            co[key] = {v: 1}
                        elif v not in co[key].keys():
                            co[key][v] = 1
                        else:
                            co[key][v] += 1
                    counter += 1
                j += 1
        
        #logging.info("Computing term cooccurrence for aspect: %s" % aspect)
        
        # Terms of interest
        tois = set(arts) | seeds
        
        # Co-occurrence
        co = {}
        for p, (sen, words) in enumerate(terms):
            words = [w if isinstance(w, int) else ("".join(w), len(w)) for w in words]
            for i, w in enumerate(words):
                if isinstance(w, int) or w[0] not in tois:
                    continue

                __search(co, tois, words, i, lambda x, y: x - y, self.__k)
                __search(co, tois, words, i, lambda x, y: x + y, self.__k)
        return co

    ###############################################################################
    # Statistics
    ###############################################################################
    def CValue(self, tinfo, contains, n):
        """
        Return top n C-Values
        """
        import math
        cvalues = {}
        for t, (size, freq) in tinfo.items():
            if t not in contains.keys():
                cvalues[t] = math.log(size) * freq
                continue

            # Multi-terms only
            nS = len(contains[t])
            freq_sum = sum(tinfo[p][1] for p in contains[t])
            cvalues[t] = math.log(size) * (freq - freq_sum / nS)

        cvalues = sorted(cvalues.items(), key=lambda x: x[1], reverse=True)
        #cvalues = [cvalues[i][0] for i in range(min(n, len(cvalues))) if cvalues[i][1] == 0]
        cvalues = [cvalues[i][0] for i in range(min(n, len(cvalues)))]
        
        logging.info("Select number ARTS with maximum CValue: %s" % len(cvalues))
        return cvalues
    
    def Max_RlogF_Term(self, arts, seeds, tinfo, co):
        """
        Return a term with maximum RlogF value and the rest of arts
        @param arts: aspect related terms
        @param tinfo: dict of term->(size, frequency)
        @param co: return value of `terms_cooccurrence`
        """
        import math

        rlogfs = []
        for art in arts:
            # If art is not in context of any aspects or seeds
            if art not in co.keys():
                continue
            
            related = [v for k, v in co[art].items() if k in seeds]
            # Not related to current seeds
            if not related:
                continue

            frq, rc = tinfo[art][1], sum(related)
            rlogfs.append((art, math.log(rc) * rc / frq))
        
        if rlogfs:
            rlogfs = sorted(rlogfs, key=lambda x: x[1], reverse=True)
            return rlogfs[0], [k for k, v in rlogfs[1:]]
        else:
            return None, []
    
    @property
    def scores(self):
        return self.__scores
    
    def term_score(self, arts):
        """
        Compute a score for each term with respect to each aspect
        @param arts: The list of (aspect, arts, ranks)
        """
        # Term ranks
        tranks = {}
        for aspect, seeds, ranks in arts:
            for term, rank, size in ranks:
                if term not in tranks.keys():
                    tranks[term] = [rank]
                else:
                    tranks[term].append(rank)

        import math
        phi = {}
        for term, rs in tranks.items():
            total = sum(rs)
            phi[term] = sum(map(lambda x: x/total * math.log(x/total), rs))

        self.__scores = {}
        for aspect, seeds, ranks in arts: # For ranks within each aspect
            ascores = {}
            for term, rank, size in ranks:
                ascores[term] = (1 - rank / size) * (1 - phi[term])
            for seed in seeds:
                ascores[seed] = 1
            
            self.__scores[aspect] = ascores

    ###############################################################################
    # Entry
    ###############################################################################
    
    def boosting(self, aspect, seeds, terms, tinfo, narts):
        
        logging.info("Boosting for aspect %s." % aspect)
            
        # Term co-occurrence
        co = self.terms_cooccurrence(aspect, seeds, terms, narts)
        
        # Return value
        import copy
        aspects, ranks, arts = set(seeds), [], copy.deepcopy(narts)
        
        # Boosting
        for i in range(self.__m):            
            art, arts = self.Max_RlogF_Term( arts, aspects, tinfo, co)

            # No more items to be found
            if not art:
                break

            ranks.append((art[0], i + 1, len(aspects)))

            aspects.add(art[0])

            # No more ART
            if len(narts) == 0:
                break
        
        logging.info("Boosting for aspect %s: %d ARTs identified." % (aspect, len(ranks)))
        
        return aspect, seeds, ranks
    
    
    def train(self, terms):
        # Collect term information
        tinfo, inclusion = self.terms_info(terms)
        
        # Top-n ARTs
        narts = self.CValue(tinfo, inclusion, self.__n)
        
        rqueue = queue.Queue()
        
        def __boosting(aspect, seeds, terms, tinfo, narts):
            ret = self.boosting(aspect, seeds, terms, tinfo, narts)
            rqueue.put((aspect, seeds, ranks))
            
        # Boost for each aspect
        threads = []
        for aspect, seeds in self.__aspects.items():
            thread = threading.Thread(target=__boosting, args=(aspect, seeds, terms, tinfo, narts))
            thread.start()
            threads.append(thread)
            
        for t in threads:
            t.join()
        
        # Collect training result
        arts = []
        while not rqueue.empty():
            arts.append(rqueue.get())
        
        self.term_score(arts)
    
    def aspect_optimal(self, sentence):
        """
        Return an optimal aspect for each sentence.
        @param aspects: A dictionary of aspect->(seeds,ranks)
        @param sentence: The sentence for aspect detection
        @return: a list of (aspect, score) for each sub-sentence of `sentence`
        """
        
        def __optimal_aspect(terms):
            arts = set()
            for ts in [t for t in terms if not isinstance(t, int)]:
                arts.add("".join(ts))
                arts |= set(arts.add(sub) for sub in self.substrings(ts, len(ts)))
            
            maxa, maxs = None, 0
            for aspect, score in self.__scores.items():                
                ascore = sum(score.get(art, 0) for art in arts)
                if ascore > maxs:
                    maxa, maxs = aspect, ascore
            
            return maxa, maxs
        
        scores = []
        for sen, terms in Extractor.single(sentence):
            aspect, score = __optimal_aspect(terms)
            if aspect:
                scores.append((aspect, score))
            
        return scores
        
        

def load(file, max_valid=None):
    """
    Load valid comments from input file
    """
    with open(file, encoding="utf8") as reader:
        import json
        rates = []
        reviews = []
        # Process each line
        counter = 0
        valid = 0
        for line in reader:
            counter += 1
            if counter == 1:
                continue
            start = line.find("^")+1
            review = json.loads(line[start:])
            if review["rate"] != -1 and str.strip(review["content"]):
                valid += 1
                rates.append(review["rate"])
                reviews.append(review["content"])
            if max_valid and valid == max_valid:
                break
        return rates, reviews

#rates, reviews = load("/home/jiakai/Data/Dianping/sentiment/reviews.txt", 10000)
#terms = Extractor.batch(reviews)

aspects = {
    "环境": set(["环境","豪华","装修","嘈杂","吵闹"]),
    "食物": set(["打折","免费","赠券","优惠","赠送"])
}
#arts = ART(aspects, 40000, 2000, 5)
#arts.train(terms)
#arts.aspect_optimal("我很喜欢的环境，因为它打折的力度很大")