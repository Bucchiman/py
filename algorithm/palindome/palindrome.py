#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# FileName: 	palindrome
# Author: 8ucchiman
# CreatedDate:  2023-02-25 13:21:25 +0900
# LastModified: 2023-02-25 16:04:24 +0900
# Reference: 8ucchiman.jp
#


import os
import sys
# import utils
# from utils import get_ml_args, get_dl_args, get_logger
# import numpy as np
# import pandas as pd


class Palindrome(object):
    @staticmethod
    def ispalindrome_fullscratch(text: str):
        """
            単純な文

        """
        isPalindrome = True
        i = 0
        j = len(text) - 1
        while isPalindrome and i <= j:
            isPalindrome &= text[i] == text[j]
            i += 1
            j -= 1
        return isPalindrome

    @staticmethod
    def ispalindrome(text: str):
        return text == text[::-1]


    @staticmethod
    def casual_longest_palindrome_substring(text: str):
        """
            https://blog.finxter.com/find-longest-palindrome-in-a-python-string-easy/
            O(N^2)
        """
        longest_palindrome = ''
        N = len(text)
        for i in range(N):
            for j in range(i+1, N+1):
                substring = text[i: j]
                if substring == substring[::-1]:
                    if len(substring) > len(longest_palindrome):
                        longest_palindrome = substring
        return longest_palindrome

    @staticmethod
    def casual_longest_palindrome_substring(text: str) -> tuple(int, list[str], list[str]):
        """
            O(N^2)
        """
        N = len(text)
        longest_length = 0
        palindromes_candidate = []
        for i in range(N):
            for j in range(i+1, N):
                substring = text[i:j]
                if Palindrome.ispalindrome(substring):
                    longest_length = max(longest_length, len(substring))
                    palindromes_candidate.append(substring)
        longest_palindromes = []
        for pc in palindromes_candidate:
            if longest_length == len(pc):
                longest_palindromes.append(pc)

        return (longest_length, longest_palindromes, palindromes_candidate)

    @staticmethod
    def center_longest_palindrome_substring(text: str):
        """
            O(N^2)
            e.g. a b c b a
                 ^
                  ^
                   ^
                    ^
                     ^
                      ^
                       ^
                        ^
                         ^   O(2N-1)
            中心からfor文でO(N)
        """
        N = len(text)
        longest_length = 0
        palindromes_candidate = []
        for center in range(2*N+1):
            left = 0
            right = 0

            if center % 2 == 0:
                left = center // 2 - 1
                right = center // 2
            else:
                left = right = center // 2

            while True:
                # left, rightがtext範囲外 or text[left] != text[right]
                if (0 > left or left >= N or 0 > right or right >= N) or text[left] != text[right]:
                    left += 1
                    right -= 1
                    break
                left -= 1
                right += 1

            length = right - left + 1
            if longest_length < length:
                longest_length = length
                palindromes_candidate.append()

        @classmethod
        def manacher_algorithm(text: str):
            """
                O(N)
                * 偶数長の場合、ダミー文字を使って奇数長にする

                文字ci(center_index)を中心とする最長の回文半径を記録(半径=(全長+1)/2)
                abaaababa
                121412321

                concept

                +------------------------------------------+
                |                    s                     |
                +------------------------------------------+

                pattern 1
                        ###########c###########    回文
                          ###c###     ###c###      回文の中に回文

                pattern 2
                        ###########c###########    回文
                              ###c###              上の回文のセンターに被った場合ｊ
                                  ###c###

                pattern 3
                        ###########c###########    回文
                      ###c###             ###c###  部分的に上の回文と被っている
                        ---                 ---    左の部分は再利用
            """
            ci = 0
            j = 0
            R = []

            while ci < len(text):
                while ci - j >= 0 and ci + j < len(text) and text[ci-j] == text[ci+j]:
                    j += 1
                R[ci] = j
                k = 1
                while ci - k >= 0 and k+R[ci-k] < j:
                    R[ci+k] = R[ci-k]
                    k += 1
                ci += k
                j -= k




def main():
    # args = utils.get_args()
    # method = getattr(utils, args.method)
    print(Palindrome.ispalindrome("madamimadam"))
    pass


if __name__ == "__main__":
    main()
