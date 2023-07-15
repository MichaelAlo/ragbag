import pandas as pd
import requests
import os
from numpy.linalg import norm
from numpy import dot
import numpy as np
import ast
from jaro import jaro_winkler_metric as jw # needs pip install jaro-winkler
from spellchecker import SpellChecker # needs pip install pyspellchecker

TABLE_PATH = "./"

class FiltersAutocomplete:

    def __init__(self):
        # set up swagger endpoint
        self.url = "https://vision.prod.booking.com/text-to-topic"

        # set up cossim
        self.th = 0.75

        # set up spellcheck
        self.spell = SpellChecker()
        self.known_words = ["on-site", "separators", "salt-water", "golfcourse", "homestay", "yukata", "wii", "checkin", "checkout", "non-smoking", "ps4", "gite", "stirrers", "bath-robe", "loungers", "ps3", "riad", "collcted", "cctv", "tv", "mini-bar", "qr", "viproom", "dj", "showerheads", "bbq", "check-out", "check-in", "ryokan", "stand-up", "blu-ray"]
        self.spell.word_frequency.load_words(self.known_words)

        # load tables
        self.top = 20
        self.sm_vcc1 = pd.read_parquet(os.path.join(TABLE_PATH, "searchmth_vcc1.parquet"))
        self.sm_vcc1_du = pd.read_parquet(os.path.join(TABLE_PATH, "searchmth_vcc1_destufi.parquet"))
        self.sm_vcc1_du_tt = pd.read_parquet(os.path.join(TABLE_PATH, "searchmth_vcc1_destufi_tt.parquet")) # removed tt as is calculated
        """
        CASE
            WHEN travel_purpose = 1 THEN "business"
            WHEN travel_purpose != 1 and nr_children = 0 and nr_adults = 2 and nr_rooms = 1 THEN "couple"
            WHEN travel_purpose != 1 and nr_children = 0 and nr_adults = 2 and nr_rooms = 2 THEN "twins"
            WHEN travel_purpose != 1 and nr_children = 0 and nr_adults > 2 THEN "group"
            WHEN travel_purpose != 1 and nr_children = 0 and nr_adults = 1 THEN "solo"
            WHEN travel_purpose != 1 and nr_children > 0 THEN "family"
            ELSE "undefined"
        END
        """
        self.facility_names = pd.read_parquet(os.path.join(TABLE_PATH, "room_hotel_facilities_hackathon.parquet"))
        # self.facility_filter_names_and_ids = self.facility_names[["facility_name_en","facility_id","filter_name_wsl"]].drop_duplicates() # can be used for mapping filter names to id and name on the website
        self.facility_names_emb = pd.read_parquet(os.path.join(TABLE_PATH, "facility_name_embedding.parquet"))

    def get_response(self, url, text):
        req_json = {"texts": [text.lower().strip()],
                    "score_threshold": 0.5,
                    "output_embeddings": True,
                    }
        response = requests.post(self.url, json=req_json)
        return response

    def cosine_similarity(self, emb1, emb2):
        emb1 = ast.literal_eval(emb1)
        cos_sim = float(dot(emb1, emb2) / (norm(emb1) * norm(emb2)))
        return cos_sim

    def jaro_winkler_search(self, text):
        self.facility_names["jw"] = self.facility_names.apply(lambda row: jw(text.lower().strip(), row.name_bis_items), axis=1)
        res = self.facility_names[self.facility_names["jw"] >= self.th][["facility_name_en","jw"]].sort_values(by="jw", ascending=False).drop_duplicates(subset="facility_name_en")
        return list(res.facility_name_en.values), list(res.jw.values)

    def spellcheck(self, text):
        words = text.lower().strip().split()
        d = {}
        misspelled = self.spell.unknown(words)
        for i, word in enumerate(misspelled):
            correction = self.spell.correction(word)
            if correction:
                d[word] = correction
        text_spellcheck = [d[w] if w in d else w for w in words]
        return ' '.join(text_spellcheck)

    def semantic_search(self, text):
        input_str = self.spellcheck(text).lower().strip()
        response = self.get_response(url=self.url, text=input_str)
        response_json = response.json()
        r = response_json["data"]
        emb = list(r[0]["embedding"])
        df = self.facility_names_emb.copy(deep=False)
        df["cossim"] = df.apply(lambda x: self.cosine_similarity(x.embeddings, emb), axis=1)
        df_filter = df[df["cossim"] >= self.th][["facility_name", "cossim"]]
        df_filter = df_filter.sort_values(by=["cossim"], ascending=False)
        return list(df_filter.facility_name.values), list(df_filter.cossim.values)

    def autocomplete(self, user_context, prefix=""):
        len_prefix = len(prefix.strip())
        sm = user_context["search_mth"]
        vc = user_context["vcc1"]

        # which precomputed (agg and sorted) table given the dict keys: search_mth (int), vcc1 (str), destufi (bigint), traveler_type (str)
        if (("traveler_type" in user_context) and ("destufi" in user_context)):
            du = user_context["destufi"]
            tt = user_context["traveler_type"]
            stats = self.sm_vcc1_du_tt.query(f"search_mth == {sm} and vcc1 == '{vc}' and dest_ufi == {du} and traveler_type == '{tt}'")[["facility_name_en","count"]]
        elif (("destufi" in user_context)):
            du = user_context["destufi"]
            stats = self.sm_vcc1_du.query(f"search_mth == {sm} and vcc1 == '{vc}' and dest_ufi == {du}")[["facility_name_en","count"]]
        else:
            stats = self.sm_vcc1.query(f"search_mth == {sm} and vcc1 == '{vc}'")[["facility_name_en","count"]]

        # return logic
        if len_prefix == 0:
            res = stats.head(self.top)
            return list(res.facility_name_en.values)
        elif len_prefix == 1:
            char = prefix.lower().strip()
            fil = self.facility_names[self.facility_names.name_bis_items.str[0] == char]["facility_name_en"].unique().tolist()
            res = stats[stats.facility_name_en.isin(fil)].sort_values(by="count", ascending=False).head(self.top)
            return list(res.facility_name_en.values)
        elif len_prefix <= 3:
            en_jw = self.jaro_winkler_search(prefix)
            en = en_jw[0]
            jw = en_jw[1]
            res = pd.DataFrame(data=zip(en, jw), columns=["facility_name_en", "jw"]).head(self.top).merge(stats[stats.facility_name_en.isin(en)][["facility_name_en", "count"]], on="facility_name_en").sort_values(by=["jw", "count"], ascending=[False, False])
            return list(res.facility_name_en.values)
        else:
            # logic: when prefix is short, it might rank jw higher, but when prefix grows, it might rank ss higher, especially if there is no fuzzy-match with jw, and if there is a match, it is a good outcome and there will be less jw matches (less candidates as string grows, narrowing down options) and so there might be room for ss to rank high too
            ss = self.semantic_search(prefix)
            jw = self.jaro_winkler_search(prefix)
            ss_name = ss[0]
            ss_cossim = ss[1]
            jw_name = jw[0]
            jw_score = jw[1]
            fil_name = ss_name + jw_name
            fil_score = ss_cossim + jw_score
            res = pd.DataFrame(data=zip(fil_name, fil_score), columns=["facility_name_en", "score"]).sort_values(by="score", ascending=False).drop_duplicates(subset="facility_name_en").head(self.top).merge(stats[stats.facility_name_en.isin(fil_name)][["facility_name_en", "count"]], on="facility_name_en").sort_values(by=["score", "count"], ascending=[False, False])
            # s = np.array(fil_score)
            # n = np.array(fil_name)
            # sort_index = (-s).argsort()[:self.top].tolist()
            return list(res.facility_name_en.values) # list(n[sort_index])
