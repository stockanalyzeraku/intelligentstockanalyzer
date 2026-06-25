from codebase.cleaning.skelton import CleanResult, SectionPattern

def _tag_page_intent(page: CleanResult) -> list[str]:
    page_number = page.page_number
    intent: list[str] = []
    for pattern, section_name in SectionPattern._SECTION_PATTERNS:
        if pattern.search(page.cleaned_text):
            intent.append(section_name)
    return intent


