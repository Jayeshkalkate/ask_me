# from django.db.models.signals import post_save
# from django.dispatch import receiver
# from django.contrib.auth.models import User
# from .models import Items

# # @receiver(post_save, sender=User)
# # def create_user_items(sender, instance, created, **kwargs):
# #     if created:
# #         Items.objects.create(user=instance)

# @receiver(post_save, sender=User)
# def create_user_items(sender, instance, created, **kwargs):
#     if created and not hasattr(instance, 'items'):
#         Items.objects.create(user=instance)
