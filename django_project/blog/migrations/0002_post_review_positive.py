# Generated by Django 3.1.2 on 2020-11-18 08:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('blog', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='review_positive',
            field=models.FloatField(default=0.0),
        ),
    ]