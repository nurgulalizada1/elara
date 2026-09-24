"""Recorded-shape responses for the research APIs (trimmed real response structures)."""

import json

import httpx

PUBMED_ESEARCH = {"esearchresult": {"count": "2", "idlist": ["38000001", "38000002"]}}
PUBMED_ESUMMARY = {"result": {
    "uids": ["38000001", "38000002"],
    "38000001": {"uid": "38000001", "title": "Single-cell RNA sequencing: a review of methods.",
                 "pubdate": "2024 Jan 5", "fulljournalname": "Nature Reviews Genetics",
                 "source": "Nat Rev Genet", "authors": [{"name": "Smith J"}, {"name": "Aliyev R"}],
                 "pubtype": ["Journal Article", "Review"],
                 "articleids": [{"idtype": "pubmed", "value": "38000001"},
                                {"idtype": "doi", "value": "10.1038/nrg.2024.1"}]},
    "38000002": {"uid": "38000002", "title": "Spatial transcriptomics of mouse brain.",
                 "pubdate": "2023", "fulljournalname": "Cell", "authors": [{"name": "Lee K"}],
                 "pubtype": ["Journal Article"],
                 "articleids": [{"idtype": "doi", "value": "10.1016/j.cell.2023.2"}]}}}
EUROPEPMC = {"resultList": {"result": [
    {"id": "38000001", "source": "MED", "pmid": "38000001", "doi": "10.1038/NRG.2024.1",
     "title": "Single-cell RNA sequencing: a review of methods.", "pubYear": "2024",
     "abstractText": "<p>We review scRNA-seq <i>methods</i>.</p>",
     "authorString": "Smith J, Aliyev R", "pubTypeList": {"pubType": ["review-article", "Review"]},
     "journalInfo": {"journal": {"title": "Nature reviews. Genetics"}}, "citedByCount": 12},
    {"id": "PPR123", "source": "PPR", "title": "A new scRNA-seq clustering method",
     "pubYear": "2025", "doi": "10.1101/2025.01.01.123", "authorString": "Doe A",
     "abstractText": "Preprint abstract."}]}}
CROSSREF = {"message": {"items": [
    {"DOI": "10.1101/2025.01.01.123", "title": ["A new scRNA-seq clustering method"],
     "type": "posted-content", "issued": {"date-parts": [[2025, 1, 1]]},
     "author": [{"given": "Ann", "family": "Doe"}], "URL": "https://doi.org/10.1101/2025.01.01.123"}]}}
S2 = {"data": [
    {"paperId": "abc", "title": "Single-cell RNA sequencing: a review of methods",
     "year": 2024, "venue": "Nature Reviews Genetics", "authors": [{"name": "J. Smith"}],
     "externalIds": {"DOI": "10.1038/nrg.2024.1", "PubMed": "38000001"},
     "publicationTypes": ["Review", "JournalArticle"], "citationCount": 40,
     "url": "https://www.semanticscholar.org/paper/abc", "abstract": "Review abstract."}]}
CLINVAR_ESEARCH = {"esearchresult": {"idlist": ["17661"]}}
CLINVAR_ESUMMARY = {"result": {"uids": ["17661"], "17661": {
    "uid": "17661", "accession": "VCV000017661", "title": "NM_007294.4(BRCA1):c.68_69del (p.Glu23fs)",
    "germline_classification": {"description": "Pathogenic",
                                "review_status": "reviewed by expert panel",
                                "last_evaluated": "2016/01/12"},
    "genes": [{"symbol": "BRCA1"}],
    "trait_set": [{"trait_name": "Hereditary breast ovarian cancer syndrome"}]}}}
GENE_ESEARCH = {"esearchresult": {"idlist": ["672"]}}
GENE_ESUMMARY = {"result": {"uids": ["672"], "672": {
    "uid": "672", "name": "BRCA1", "description": "BRCA1 DNA repair associated",
    "summary": "This gene encodes a nuclear phosphoprotein...", "chromosome": "17",
    "maplocation": "17q21.31", "organism": {"scientificname": "Homo sapiens"}}}}
ENSEMBL_GENE = {"id": "ENSG00000012048", "display_name": "BRCA1", "seq_region_name": "17",
                "start": 43044292, "end": 43170245, "strand": -1, "biotype": "protein_coding",
                "description": "BRCA1 DNA repair associated [Source:HGNC Symbol;Acc:HGNC:1100]"}
ENSEMBL_VAR = {"name": "rs80357906", "MAF": None, "most_severe_consequence": "frameshift_variant",
               "mappings": [{"location": "17:43057063-43057062", "allele_string": "-/G"}],
               "clinical_significance": ["pathogenic"]}
GNOMAD = {"data": {"variant": {"variant_id": "17-43057062-T-TG", "rsids": ["rs80357906"],
                               "chrom": "17", "pos": 43057062, "ref": "T", "alt": "TG",
                               "exome": {"ac": 3, "an": 1461000}, "genome": None}}}


def handler(overrides: dict | None = None, calls: list | None = None):
    """MockTransport handler routing by URL. overrides: {substring: Response|Exception}."""
    overrides = overrides or {}

    def h(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if calls is not None:
            calls.append(url)
        for key, val in overrides.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val
        db = req.url.params.get("db")
        if "esearch" in url:
            return httpx.Response(200, json={"pubmed": PUBMED_ESEARCH, "clinvar": CLINVAR_ESEARCH,
                                             "gene": GENE_ESEARCH}[db])
        if "esummary" in url:
            return httpx.Response(200, json={"pubmed": PUBMED_ESUMMARY,
                                             "clinvar": CLINVAR_ESUMMARY,
                                             "gene": GENE_ESUMMARY}[db])
        if "europepmc" in url:
            return httpx.Response(200, json=EUROPEPMC)
        if "crossref" in url:
            return httpx.Response(200, json=CROSSREF)
        if "semanticscholar" in url:
            return httpx.Response(200, json=S2)
        if "lookup/symbol" in url:
            return httpx.Response(200, json=ENSEMBL_GENE)
        if "variation/human" in url:
            return httpx.Response(200, json=ENSEMBL_VAR)
        if "gnomad" in url:
            body = json.loads(req.content)
            assert "variant(" in body["query"]
            return httpx.Response(200, json=GNOMAD)
        return httpx.Response(404, json={"error": "not mocked: " + url})

    return h
