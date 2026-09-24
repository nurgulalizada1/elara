"""User-facing strings for deterministic (non-LLM) responses, in az / en / tr."""

from __future__ import annotations

MESSAGES: dict[str, dict[str, str]] = {
    "greeting": {"az": "Salam! Necəsən?", "en": "Hi! How are you?",
                 "tr": "Merhaba! Nasılsın?"},
    "how_are_you": {"az": "Yaxşıyam, sağ ol! Sənə necə kömək edə bilərəm?",
                    "en": "I'm good, thanks! What can I do for you?",
                    "tr": "İyiyim, teşekkürler! Nasıl yardımcı olabilirim?"},
    "thanks": {"az": "Dəyməz!", "en": "You're welcome!", "tr": "Rica ederim!"},
    "goodbye": {"az": "Sağ ol, görüşənədək!", "en": "Bye, talk soon!",
                "tr": "Görüşürüz!"},
    "calc_result": {"az": "{expr} = {value}", "en": "{expr} = {value}", "tr": "{expr} = {value}"},
    "calc_error": {"az": "Bunu hesablaya bilmədim: {error}",
                   "en": "I couldn't calculate that: {error}",
                   "tr": "Bunu hesaplayamadım: {error}"},
    "time_now": {"az": "İndi saat {time}, {date}.", "en": "It's {time}, {date}.",
                 "tr": "Şu an saat {time}, {date}."},
    "memory_saved": {"az": "Yadda saxladım: {content}", "en": "Got it, I'll remember: {content}",
                     "tr": "Aklımda tutacağım: {content}"},
    "memory_updated": {"az": "Yaddaşı yenilədim: {content} (əvvəlki: {old})",
                       "en": "Updated: {content} (was: {old})",
                       "tr": "Güncelledim: {content} (önceki: {old})"},
    "memory_duplicate": {"az": "Bunu artıq bilirəm: {content}",
                         "en": "I already know that: {content}",
                         "tr": "Bunu zaten biliyorum: {content}"},
    "memory_rejected": {"az": "Bunu yadda saxlamadım: {reason}",
                        "en": "I didn't save that: {reason}",
                        "tr": "Bunu kaydetmedim: {reason}"},
    "memory_empty": {"az": "Hələ heç nə yadda saxlamamışam.",
                     "en": "I haven't saved anything about you yet.",
                     "tr": "Henüz hakkında bir şey kaydetmedim."},
    "memory_list_header": {"az": "Yadda saxladıqlarım:", "en": "Here's what I remember:",
                           "tr": "Hatırladıklarım:"},
    "memory_not_found": {"az": "Bu barədə yaddaşımda heç nə yoxdur.",
                         "en": "I don't have anything saved about that.",
                         "tr": "Bununla ilgili kayıtlı bir şey yok."},
    "memory_deleted": {"az": "Unutdum: {content}", "en": "Forgotten: {content}",
                       "tr": "Unuttum: {content}"},
    "memory_ambiguous": {
        "az": "Bir neçə uyğun qeyd var. Hansını silim? (/forget <id>)\n{items}",
        "en": "Several memories match. Which one should I delete? (/forget <id>)\n{items}",
        "tr": "Birden fazla kayıt eşleşiyor. Hangisini sileyim? (/forget <id>)\n{items}"},
    "confirm_request": {
        "az": "Bu əməliyyat təsdiq tələb edir: {action}\nSəbəb: {reason}\nDavam edim? (bəli / xeyr)",
        "en": "This needs your confirmation: {action}\nWhy: {reason}\nProceed? (yes / no)",
        "tr": "Bu işlem onay gerektiriyor: {action}\nNeden: {reason}\nDevam edeyim mi? (evet / hayır)"},
    "confirm_cancelled": {"az": "Ləğv etdim.", "en": "Cancelled.", "tr": "İptal ettim."},
    "confirm_expired": {"az": "Təsdiq müddəti bitdi, əməliyyat icra olunmadı.",
                        "en": "That confirmation expired; nothing was done.",
                        "tr": "Onay süresi doldu; hiçbir şey yapılmadı."},
    "tool_failed": {"az": "Alınmadı ({tool}): {error}", "en": "That failed ({tool}): {error}",
                    "tr": "Başarısız oldu ({tool}): {error}"},
    "tool_denied": {"az": "Buna icazəm yoxdur ({tool}): {error}",
                    "en": "I'm not allowed to do that ({tool}): {error}",
                    "tr": "Buna iznim yok ({tool}): {error}"},
    "llm_unavailable": {
        "az": "Bunun üçün dil modelinə ehtiyacım var, amma model hazırda əlçatan deyil ({error}). "
              "Sadə əmrlər (hesablama, yaddaş, fayllar, tədqiqat axtarışı) yenə də işləyir.",
        "en": "I need a language model for that, but it's unavailable right now ({error}). "
              "Simple commands (math, memory, files, research search) still work.",
        "tr": "Bunun için dil modeline ihtiyacım var ama şu an erişilemiyor ({error}). "
              "Basit komutlar (hesaplama, hafıza, dosyalar, araştırma) yine çalışır."},
    "step_limit": {"az": "Tapşırığı ayrılmış addım sayında bitirə bilmədim.",
                   "en": "I couldn't finish within the step limit.",
                   "tr": "Görevi adım sınırı içinde bitiremedim."},
    "refusal": {"az": "Bu sorğuya cavab verə bilmirəm.", "en": "I can't help with that request.",
                "tr": "Bu isteğe yardımcı olamam."},
    "research_none": {
        "az": "“{query}” üzrə heç bir mənbədə nəticə tapmadım.",
        "en": "I found no results for “{query}” in any source.",
        "tr": "“{query}” için hiçbir kaynakta sonuç bulamadım."},
    "research_header": {"az": "“{query}” üzrə tapdıqlarım:",
                        "en": "Here's what I found for “{query}”:",
                        "tr": "“{query}” için bulduklarım:"},
    "research_failed_sources": {"az": "Əlçatan olmayan mənbələr: {sources}",
                                "en": "Sources that failed: {sources}",
                                "tr": "Erişilemeyen kaynaklar: {sources}"},
    "research_no_llm_note": {
        "az": "(Dil modeli olmadığı üçün xülasə yazmadım; yuxarıdakılar birbaşa mənbə məlumatlarıdır.)",
        "en": "(No language model available, so no synthesis; the list above is raw source metadata.)",
        "tr": "(Dil modeli olmadığı için özet yazmadım; yukarıdakiler doğrudan kaynak verileridir.)"},
    "reference_unresolved": {
        "az": "Hansı nəticəni nəzərdə tutduğunu tapa bilmədim.",
        "en": "I couldn't tell which earlier result you mean.",
        "tr": "Hangi önceki sonucu kastettiğini anlayamadım."},
    "input_too_long": {"az": "Mesaj çox uzundur ({n} simvol, maksimum {max}).",
                       "en": "That message is too long ({n} characters, max {max}).",
                       "tr": "Mesaj çok uzun ({n} karakter, en fazla {max})."},
    "empty_input": {"az": "Nəsə yazmaq istədin?", "en": "Did you want to say something?",
                    "tr": "Bir şey mi yazmak istedin?"},
    "db_error": {"az": "Yerli verilənlər bazasında xəta baş verdi: {error}",
                 "en": "A local database error occurred: {error}",
                 "tr": "Yerel veritabanında hata oluştu: {error}"},
}


def t(key: str, lang: str, **kwargs) -> str:
    table = MESSAGES[key]
    template = table.get(lang) or table["en"]
    return template.format(**kwargs)
