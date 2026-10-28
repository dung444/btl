"""Tìm sản phẩm bằng từ khóa tiếng Việt hoặc tiếng Anh."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


# Mỗi từ khóa tiếng Việt được đổi thành các tên loại hàng thường gặp trong dữ liệu.
# Giữ từ khóa tiếng Anh ở cuối để truy vấn vẫn tìm được tên sản phẩm gốc.
VIETNAMESE_TERMS = {
    "ao so mi": ["shirt"],
    "ao thun": ["tshirt", "t-shirt", "tee"],
    "ao khoac": ["jacket", "sweatshirt", "hoodie"],
    "dam": ["dress"],
    "vay": ["dress", "skirt"],
    "ao": ["shirt", "top", "tshirt", "tee", "kurta", "jacket"],
    "quan": ["trouser", "jean", "pant", "short"],
    "giay": ["shoe", "sneaker", "heel", "flat"],
    "dep": ["sandal", "flip flop", "slipper"],
    "tui": ["bag", "handbag", "backpack"],
    "dong ho": ["watch"],
    "kinh": ["sunglass", "eyewear", "spectacle"],
    "do": ["red"],
    "xanh": ["blue", "green"],
    "den": ["black"],
    "trang": ["white"],
    "hong": ["pink"],
    "vang": ["yellow", "gold"],
    "nam": ["men"],
    "nu": ["women", "girls"],
}


def normalize_query(value: str) -> str:
    """Bỏ dấu tiếng Việt, đưa về chữ thường và gộp khoảng trắng."""
    value = unicodedata.normalize("NFD", value.casefold())
    value = "".join(character for character in value if unicodedata.category(character) != "Mn")
    value = value.replace("đ", "d")
    return " ".join(value.split())


def search_products(products: pd.DataFrame, query: str) -> pd.DataFrame:
    """Tìm theo tên và thuộc tính; các nhóm từ trong truy vấn phải cùng khớp."""
    normalized = normalize_query(query)
    if not normalized:
        return products

    # Tên hàng và các thuộc tính được tìm đồng thời để truy vấn như “váy đỏ” hoạt động.
    searchable_columns = [
        "productDisplayName", "articleType", "subCategory", "masterCategory",
        "baseColour", "gender", "usage", "season",
    ]
    searchable = products[searchable_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    words = normalized.split()
    groups: list[list[str]] = []
    while words:
        # Ưu tiên cụm dài trước để “áo sơ mi” không bị tách thành ba từ riêng.
        found = False
        for length in range(min(3, len(words)), 0, -1):
            phrase = " ".join(words[:length])
            if phrase in VIETNAMESE_TERMS:
                groups.append(VIETNAMESE_TERMS[phrase])
                words = words[length:]
                found = True
                break
        if not found:
            groups.append([words.pop(0)])

    result_mask = pd.Series(True, index=products.index)
    for alternatives in groups:
        pattern = "|".join(re.escape(term) for term in alternatives)
        result_mask &= searchable.str.contains(pattern, regex=True, na=False)
    return products[result_mask]
