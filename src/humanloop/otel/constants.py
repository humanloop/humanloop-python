# Attribute name prefix on Humanloop spans for file-related attributes + path
HL_FILE_OT_KEY = "humanloop.file"
# Attribute name prefix on Humanloop spans for log-related attributes
HL_LOG_OT_KEY = "humanloop.log"
# Attribute name prefix on Humanloop spans for trace metadata
HL_TRACE_METADATA_KEY = "humanloop.flow.metadata"
# OTel does not allow falsy values for top-level attributes e.g. foo
# and None only on nested attributes e.g. foo.bar
HL_OT_EMPTY_VALUE = "EMPTY"
