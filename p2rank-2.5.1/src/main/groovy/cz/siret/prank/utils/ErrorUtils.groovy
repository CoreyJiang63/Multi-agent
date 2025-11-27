package cz.siret.prank.utils

import org.apache.commons.lang3.exception.ExceptionUtils

/**
 *
 */
class ErrorUtils {

    static List<String> getAllCauseMessages(Throwable throwable) {
        List<String> messages = new ArrayList<>()
        while (throwable != null) {
            messages.add(throwable.getMessage())
            throwable = throwable.getCause()
        }
        return messages
    }

    static List<String> getAllCauseMessagesWithClasses(Throwable throwable) {
        List<String> messages = new ArrayList<>()
        while (throwable != null) {
            messages.add("$throwable.class.simpleName: $throwable.message")
            throwable = throwable.getCause()
        }
        return messages
    }

    static String getRootCauseMessage(Throwable throwable) {
        return ExceptionUtils.getRootCauseMessage(throwable)
    }

    public static String stackTraceToString(Throwable throwable) {
        return ExceptionUtils.getStackTrace(throwable)
    }

    /**
     * Converts the full stack trace of a Throwable, including all its causes,
     * into a String.
     *
     * @param throwable the Throwable to process
     * @return a String containing the full stack trace with causes
     */
    public static String fullStackTraceToString(Throwable throwable) {
        if (throwable == null) {
            return ""
        }

        StringBuilder sb = new StringBuilder()
        boolean first = true
        ;
        // Process the main throwable and all its causes
        for (Throwable current = throwable; current != null; current = current.getCause()) {
            // Add appropriate prefix for causes
            if (first) {
                first = false
            } else {
                sb.append("Caused by: ")
            }

            // Add exception class and message
            sb.append(current.toString()).append("\n")

            // Add stack trace elements
            for (StackTraceElement element : current.getStackTrace()) {
                sb.append("\tat ").append(element).append("\n")
            }
        }

        return sb.toString();
    }


}
