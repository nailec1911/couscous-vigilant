##
## EPITECH PROJECT, 2024
## B-CNA-500-BDX-5-1-cryptography-adam.de-lacheisserie-levy
## File description:
## Makefile
##

NAME_ANAL	=	my_torch_analyzer
NAME_GEN	=	my_torch_generator

all: $(NAME_ANAL) $(NAME_GEN)

$(NAME_ANAL):
	cp my_torch_analyzer.py my_torch_analyzer
	chmod +x my_torch_analyzer

$(NAME_GEN):
	cp my_torch_generator.py my_torch_generator
	chmod +x my_torch_generator



fclean:
	rm -f $(NAME_ANAL)
	rm -f $(NAME_GEN)

re: fclean all


.PHONY: all $(NAME_ANAL) $(NAME_GEN)
