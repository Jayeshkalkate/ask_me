from django.db import migrations
import core.crypto_fields


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0002_sharedlink"),
    ]

    operations = [
        # These swap the Python field class only (JSONField/TextField ->
        # transparently-encrypted subclasses in core/crypto_fields.py). The
        # underlying database column type is unchanged, so this migration
        # does not touch existing data - it just changes how Django
        # serializes/deserializes these columns from here on.
        migrations.AlterField(
            model_name="document",
            name="extracted_text",
            field=core.crypto_fields.EncryptedTextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="document",
            name="extracted_data",
            field=core.crypto_fields.EncryptedJSONField(blank=True, default=dict),
        ),
        migrations.AlterField(
            model_name="document",
            name="user_edited_data",
            field=core.crypto_fields.EncryptedJSONField(blank=True, default=dict),
        ),
        migrations.AlterField(
            model_name="document",
            name="ai_extracted_json",
            field=core.crypto_fields.EncryptedJSONField(blank=True, default=dict),
        ),
    ]
