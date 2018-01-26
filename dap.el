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
	     '("tf" "#+BEGIN_SRC tf :results output drawer org\nimport tensorflow as tf\n?\n#+END_SRC"
	       "<src lang=\"python\">\n?\n</src>"))

(defface org-block-tf
  `((t (:background "LightSalmon1")))
  "Face for tensorflow python blocks")


(provide 'dap)

;;; dap.el ends here
