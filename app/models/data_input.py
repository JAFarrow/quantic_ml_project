from marshmallow import Schema, fields, validate, ValidationError, RAISE, pre_load

def _empty_to_none(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


class PredictionRowSchema(Schema):
    class Meta:
        unknown = RAISE

    BaseOfCode = fields.Integer(required=True)
    BaseOfData = fields.Integer(required=True)
    Characteristics = fields.Integer(required=True)
    DllCharacteristics = fields.Integer(required=True)
    Entropy = fields.Float(required=True)
    FileAlignment = fields.Integer(required=True)
    FirstSeenDate = fields.Date(required=True)
    Identify = fields.String(required=True, allow_none=True)
    ImageBase = fields.Integer(required=True)
    ImportedDlls = fields.String(required=True)
    ImportedSymbols = fields.String(required=True)
    Machine = fields.Integer(required=True)
    Magic = fields.Integer(required=False, allow_none=True)
    NumberOfRvaAndSizes = fields.Integer(required=True)
    NumberOfSections = fields.Integer(required=True)
    NumberOfSymbols = fields.Integer(required=True)
    PE_TYPE = fields.Integer(required=False, allow_none=True)
    PointerToSymbolTable = fields.Integer(required=True)
    SHA1 = fields.String(required=False, allow_none=True)
    Size = fields.Integer(required=True)
    SizeOfCode = fields.Integer(required=True)
    SizeOfHeaders = fields.Integer(required=True)
    SizeOfImage = fields.Integer(required=True)
    SizeOfInitializedData = fields.Integer(required=True)
    SizeOfOptionalHeader = fields.Integer(required=False, allow_none=True)
    SizeOfUninitializedData = fields.Integer(required=True)
    TimeDateStamp = fields.Integer(required=True)
    Label = fields.Integer(required=False, allow_none=True, validate=validate.OneOf([0, 1]))

    @pre_load
    def normalize_input(self, data, **kwargs):
        if not isinstance(data, dict):
            raise ValidationError("Each row must be an object/dictionary.")
        normalized = {}
        for key, value in data.items():
            normalized[key] = _empty_to_none(value)
        return normalized


class PredictionBatchSchema(Schema):
    rows = fields.List(
        fields.Nested(PredictionRowSchema),
        required=True,
        validate=validate.Length(min=1, max=5000)
    )