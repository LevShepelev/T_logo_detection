# позитивные — строгие (по умолчанию)
POS_STRICT = [
    "white heater shield (flat top, pointed bottom) containing a single black Latin capital letter T centered; monochrome; no other letters; no words; not circular; not oval",
    "minimalist heater-shield icon with only one bold Latin 'T' centered; high contrast; no text; not a circle; not an oval",
    "badge/crest shaped as a heater shield with a single serif capital T in the middle; no surrounding text; no extra symbols",
    "щит типа 'heater' (плоский верх, заострённый низ) с одной латинской заглавной буквой T по центру; монохромно; без слов; не круг; не овал",
]

# сбалансированные (если нужно поднять recall)
POS_BALANCED = [
    "Latin capital T centered inside a heater shield emblem; no other letters; monochrome; not circular",
    "stylized letter T inside a shield icon; no surrounding text; single letter",
    "буква T в щите (эмблема), одна буква, без текста вокруг",
]

# негативные — для вычитания
NEG_DEFAULT = [
    # VK / монограммы
    "VK monogram logo; letters V and K together; white VK mark",
    "monogram made of multiple letters without a shield",
    # яйцо МТС / круги / овалы
    "egg-shaped icon inside a rounded square; MTS logo",
    "solid circle icon; solid oval icon; round badge without letters",
    # РСХБ
    "round emblem with wheat and a key; bank logo; circular crest",
    # просто текст / одиночная буква без щита
    "standalone letter T without a shield; word or text containing letter T",
    "слово Тинькофф; текст с буквой Т; буква Т без щита",
    "круглая эмблема; овальная иконка; монограмма из нескольких букв",
]

SETS = {
    "strict": POS_STRICT,
    "balanced": POS_BALANCED,
    "neg": NEG_DEFAULT,
}
