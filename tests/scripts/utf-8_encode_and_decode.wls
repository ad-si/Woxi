utf = ToCharacterCode[ToString["AöЖ€", CharacterEncoding -> "UTF8"]]
ToCharacterCode[FromCharacterCode[utf, "UTF8"]]
