(deftemplate MAIN::class
   (slot name)
   (multislot attributes)
   (multislot operations))

(deftemplate MAIN::attribute
   (slot id)
   (slot class-name)
   (slot name)
   (slot visibility)
   (slot type))

(deftemplate MAIN::operation
   (slot id)
   (slot class-name)
   (slot name)
   (slot visibility)
   (slot type))

(deftemplate MAIN::dependency
   (slot client)
   (slot supplier))

(deftemplate MAIN::generalization
   (slot parent)
   (slot child))

(deftemplate MAIN::directedAssociation
   (slot source)
   (slot target)
   (slot multiplicity1)
   (slot multiplicity2))

(deftemplate MAIN::association
   (slot source)
   (slot target)
   (slot multiplicity1)
   (slot multiplicity2))

(deftemplate MAIN::composition
   (slot whole)
   (slot part)
   (slot multiplicity))

(deftemplate MAIN::aggregation
   (slot whole)
   (slot part)
   (slot multiplicity))

(deffacts MAIN::initial-facts
   (attribute (id attr1) (class-name ClaseHijossL) (name gg) (visibility public) (type "string"))
   (operation (id op1) (class-name ClaseHijossL) (name Correr) (visibility public) (type "int"))
   (class (name ClaseHijossL) (attributes attr1) (operations op1)))

(defrule MAIN::generate-java-code
   ?class <- (class (name ?class-name) (attributes $?attributes) (operations $?operations))
   (generalization (parent ?class-name) (child ?x))
   =>
   (printout t "// Java code for class " ?class-name crlf)
   (printout t "public class " ?class-name " extends " ?x " {" crlf)
   (do-for-all-facts ((?attr attribute))
      (and (member$ (fact-slot-value ?attr id) $?attributes) (eq (fact-slot-value ?attr class-name) ?class-name))
      (bind ?visibility (fact-slot-value ?attr visibility))
      (bind ?type (fact-slot-value ?attr type))
      (bind ?name (fact-slot-value ?attr name))
      (printout t "   " ?visibility " " ?type " " ?name ";" crlf))
   (do-for-all-facts ((?op operation))
      (and (member$ (fact-slot-value ?op id) $?operations) (eq (fact-slot-value ?op class-name) ?class-name))
      (bind ?visibility (fact-slot-value ?op visibility))
      (bind ?type (fact-slot-value ?op type))
      (bind ?name (fact-slot-value ?op name))
      (printout t "   " ?visibility " " ?type " " ?name "()" " {" crlf "      // method body" crlf "   }" crlf))
   (printout t "}" crlf crlf))

(defrule MAIN::generate-java-code-no-inheritance
   ?class <- (class (name ?class-name) (attributes $?attributes) (operations $?operations))
   (not (generalization (parent ?class-name)))
   =>
   (printout t "// Java code for class " ?class-name crlf)
   (printout t "public class " ?class-name " {" crlf)
   (do-for-all-facts ((?attr attribute))
      (and (member$ (fact-slot-value ?attr id) $?attributes) (eq (fact-slot-value ?attr class-name) ?class-name))
      (bind ?visibility (fact-slot-value ?attr visibility))
      (bind ?type (fact-slot-value ?attr type))
      (bind ?name (fact-slot-value ?attr name))
      (printout t "   " ?visibility " " ?type " " ?name ";" crlf))
   (do-for-all-facts ((?op operation))
      (and (member$ (fact-slot-value ?op id) $?operations) (eq (fact-slot-value ?op class-name) ?class-name))
      (bind ?visibility (fact-slot-value ?op visibility))
      (bind ?type (fact-slot-value ?op type))
      (bind ?name (fact-slot-value ?op name))
      (printout t "   " ?visibility " " ?type " " ?name "()" " {" crlf "      // method body" crlf "   }" crlf))
   (printout t "}" crlf crlf))

