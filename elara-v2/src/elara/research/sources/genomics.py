"""Genomics databases: Ensembl REST and gnomAD GraphQL."""

from __future__ import annotations

from elara.core.errors import SourceError
from elara.research.models import EvidenceType, Record, ResearchIntent, ResearchPlan
from elara.research.sources.base import ResearchSource

ENSEMBL = "https://rest.ensembl.org"


class EnsemblSource(ResearchSource):
    name = "ensembl"
    description = "Ensembl gene and variant annotation (human, GRCh38)"
    intents = {ResearchIntent.GENE, ResearchIntent.VARIANT}
    health_url = f"{ENSEMBL}/info/ping?content-type=application/json"

    def applicable(self, plan: ResearchPlan) -> bool:
        return bool(plan.identifiers.get("gene") or plan.identifiers.get("rsid"))

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        params = {"content-type": "application/json"}
        if rsid := plan.identifiers.get("rsid"):
            d = await self.http.get_json(f"{ENSEMBL}/variation/human/{rsid}", params=params,
                                         source=self.name)
            if not isinstance(d, dict) or "name" not in d:
                return []
            maps = d.get("mappings") or []
            loc = ", ".join(f"{m.get('location')} ({m.get('allele_string')})" for m in maps[:3])
            sig = ", ".join(d.get("clinical_significance") or []) or "n/a"
            return [Record(
                source=self.name, source_id=d["name"], title=f"Ensembl variant {d['name']}",
                url=f"https://www.ensembl.org/Homo_sapiens/Variation/Explore?v={d['name']}",
                evidence_type=EvidenceType.DATABASE_RECORD,
                abstract=(f"Location: {loc or 'n/a'}. Most severe consequence: "
                          f"{d.get('most_severe_consequence', 'n/a')}. Minor allele "
                          f"{d.get('minor_allele') or 'n/a'} (global MAF {d.get('MAF', 'n/a')}). "
                          f"Clinical significance (as reported to Ensembl): {sig}."),
                extra={"mappings": maps[:3], "maf": d.get("MAF"),
                       "consequence": d.get("most_severe_consequence")})]
        gene = plan.identifiers["gene"]
        d = await self.http.get_json(f"{ENSEMBL}/lookup/symbol/homo_sapiens/{gene}",
                                     params=params, source=self.name)
        if not isinstance(d, dict) or "id" not in d:
            return []
        return [Record(
            source=self.name, source_id=d["id"],
            title=f"{d.get('display_name', gene)} ({d['id']})",
            url=f"https://www.ensembl.org/Homo_sapiens/Gene/Summary?g={d['id']}",
            evidence_type=EvidenceType.DATABASE_RECORD,
            abstract=(f"{d.get('description') or ''} Location: chr{d.get('seq_region_name')}:"
                      f"{d.get('start')}-{d.get('end')} (strand {d.get('strand')}), biotype "
                      f"{d.get('biotype')}, assembly {d.get('assembly_name', 'GRCh38')}.").strip(),
            extra={k: d.get(k) for k in ("seq_region_name", "start", "end", "strand",
                                         "biotype")})]


_GNOMAD_QUERY = """
query ElaraVariant($variantId: String, $rsid: String, $dataset: DatasetId!) {
  variant(variantId: $variantId, rsid: $rsid, dataset: $dataset) {
    variant_id
    rsids
    chrom
    pos
    ref
    alt
    exome { ac an }
    genome { ac an }
  }
}"""


class GnomADSource(ResearchSource):
    name = "gnomad"
    description = "gnomAD population allele frequencies (v4)"
    intents = {ResearchIntent.VARIANT}
    health_url = None  # GraphQL endpoint only accepts POST
    dataset = "gnomad_r4"

    def applicable(self, plan: ResearchPlan) -> bool:
        return bool(plan.identifiers.get("variant_id") or plan.identifiers.get("rsid"))

    async def search(self, plan: ResearchPlan, limit: int) -> list[Record]:
        variables = {"dataset": self.dataset, "variantId": plan.identifiers.get("variant_id"),
                     "rsid": None if plan.identifiers.get("variant_id")
                     else plan.identifiers.get("rsid")}
        data = await self.http.post_json("https://gnomad.broadinstitute.org/api",
                                         body={"query": _GNOMAD_QUERY, "variables": variables},
                                         source=self.name)
        if data.get("errors") and not (data.get("data") or {}).get("variant"):
            msg = data["errors"][0].get("message", "unknown error")
            if "not found" in msg.lower():
                return []
            raise SourceError(f"gnomad: {msg[:200]}")
        v = (data.get("data") or {}).get("variant")
        if not v:
            return []
        parts, total_ac, total_an = [], 0, 0
        for label in ("exome", "genome"):
            s = v.get(label)
            if s and s.get("an"):
                parts.append(f"{label}: AC={s['ac']}, AN={s['an']}, AF={s['ac'] / s['an']:.3g}")
                total_ac += s["ac"]
                total_an += s["an"]
        af = total_ac / total_an if total_an else None
        return [Record(
            source=self.name, source_id=v["variant_id"],
            title=f"gnomAD v4 variant {v['variant_id']}"
                  + (f" ({', '.join(v.get('rsids') or [])})" if v.get("rsids") else ""),
            url=f"https://gnomad.broadinstitute.org/variant/{v['variant_id']}?dataset={self.dataset}",
            evidence_type=EvidenceType.DATABASE_RECORD,
            abstract=("Population frequencies — " + ("; ".join(parts) or "no frequency data")
                      + (f". Combined AF={af:.3g}." if af is not None else ".")),
            extra={"allele_frequency": af, "exome": v.get("exome"), "genome": v.get("genome")})]
