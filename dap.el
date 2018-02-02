;;; dap.el --- Emacs setup for dap.
;; * Convenience function for tensorflow blocks in org-mode

;;; Commentary:
;;

;;; Code:

(defalias 'org-babel-execute:tf 'org-babel-execute:python)
(defalias 'org-babel-prep-session:tf 'org-babel-prep-session:python)
(defalias 'org-babel-tf-initiate-session 'org-babel-python-initiate-session)

(add-to-list 'org-src-lang-modes '("tf" . python))
(add-to-list 'org-latex-minted-langs '(tf  "python"))

(setq org-src-block-faces '(("tf" (:background "#EEE2FF"))))

(add-to-list 'org-structure-template-alist
	     '("tf" "#+BEGIN_SRC tf :results output drawer org\nimport tensorflow as tf\n?\n\nwith tf.Session():\n    #+END_SRC"
	       "<src lang=\"python\">\n?\n</src>"))

(require 'color)

(defface org-block-tf
  `((t (:background ,(color-lighten-name "LightSalmon1" 0.50))))
  "Face for tensorflow python blocks")


(defun dap-insert-header ()
  (interactive)
  (goto-char (point-min))
  (insert "# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"))


(defun dap-run-test-in-shell ()
  "Run the test that point is in in a shell.
Assumes the test is in a class in a file in dap/tests/"
  (interactive)
  (let (cmd file class method)
    (save-excursion
      (re-search-backward "^class \\(.*\\)(")
      (setq class (s-trim (match-string 1))))
    (save-excursion
      (re-search-backward "def \\(.*\\)(")
      (setq method (match-string 1)))
    (setq file (file-name-base (buffer-file-name)))
    (setq cmd (format "python -m unittest dap.tests.%s.%s.%s" file class method))
    (message (shell-command-to-string cmd))))

(provide 'dap)

;;; dap.el ends here
